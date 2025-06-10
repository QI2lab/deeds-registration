import unittest
import numpy as np
import time

from ..registration import (
    registration,
    registration_fields,
    registration_imwarp_fields,
    deeds,
    deeds_fields,
    deeds_imwarp_fields,
)


class TestRegistration(unittest.TestCase):
    def setUp(self):
        self.fixed = np.zeros((8, 8, 8), dtype=np.float32)
        self.fixed[2:6, 2:6, 2:6] = 1.0

    def test_registration_translation(self):
        shift = (1, -2, 3)
        moving = np.roll(self.fixed, shift, axis=(0, 1, 2))
        moved = registration(self.fixed, moving, levels=3)
        np.testing.assert_allclose(moved, self.fixed, atol=1e-4)

    def test_flow_values(self):
        shift = (0, 1, -1)
        moving = np.roll(self.fixed, shift, axis=(0, 1, 2))
        vz, vy, vx = registration_fields(self.fixed, moving, levels=3)
        self.assertAlmostEqual(float(vz.mean()), -shift[0], places=1)
        self.assertAlmostEqual(float(vy.mean()), -shift[1], places=1)
        self.assertAlmostEqual(float(vx.mean()), -shift[2], places=1)

    def test_imwarp_translation(self):
        shift = (-1, 0, 2)
        moving = np.roll(self.fixed, shift, axis=(0, 1, 2))
        moved, vz, vy, vx = registration_imwarp_fields(self.fixed, moving, levels=3)
        np.testing.assert_allclose(moved, self.fixed, atol=1e-4)
        self.assertEqual(vz.shape, self.fixed.shape)
        self.assertEqual(vy.shape, self.fixed.shape)
        self.assertEqual(vx.shape, self.fixed.shape)

    def test_deeds_wrappers(self):
        shift = (2, -1, 0)
        moving = np.roll(self.fixed, shift, axis=(0, 1, 2))
        moved = deeds(self.fixed, moving, levels=3)
        np.testing.assert_allclose(moved, self.fixed, atol=1e-4)
        vz, vy, vx = deeds_fields(self.fixed, moving, levels=3)
        self.assertAlmostEqual(float(vz.mean()), -shift[0], places=1)
        self.assertAlmostEqual(float(vy.mean()), -shift[1], places=1)
        self.assertAlmostEqual(float(vx.mean()), -shift[2], places=1)
        moved2, vz2, vy2, vx2 = deeds_imwarp_fields(self.fixed, moving, levels=3)
        np.testing.assert_allclose(moved2, self.fixed, atol=1e-4)
        np.testing.assert_allclose(vz, vz2)
        np.testing.assert_allclose(vy, vy2)
        np.testing.assert_allclose(vx, vx2)

    def test_gpu_cpu_parity(self):
        from .. import gpu_registration as gr
        if not gr._gpu_ok:
            self.skipTest("GPU not available")
        img = np.random.rand(4, 4, 4).astype(np.float32)
        cpu_shift = gr._imshift_cpu(img, 1, -1, 0)
        gpu_shift = gr._imshift_gpu(gr.cp.asarray(img), 1, -1, 0)
        np.testing.assert_allclose(cpu_shift, gr.cp.asnumpy(gpu_shift))

        cpu_desc = gr._descriptor_cpu(img, qs=1)
        gpu_desc = gr._descriptor_gpu(gr.cp.asarray(img), qs=1)
        np.testing.assert_array_equal(cpu_desc, gr.cp.asnumpy(gpu_desc))

        a = (np.random.rand(10) * 1e6).astype(np.uint64)
        b = (np.random.rand(10) * 1e6).astype(np.uint64)
        cpu_ham = gr.hamming_distance_cpu(a, b)
        gpu_ham = gr.hamming_distance_gpu(gr.cp.asarray(a), gr.cp.asarray(b))
        np.testing.assert_allclose(cpu_ham, gr.cp.asnumpy(gpu_ham))

        # parity for image warping
        flow_u = gr.cp.random.uniform(-1, 1, size=img.shape).astype(gr.cp.float32)
        flow_v = gr.cp.random.uniform(-1, 1, size=img.shape).astype(gr.cp.float32)
        flow_w = gr.cp.random.uniform(-1, 1, size=img.shape).astype(gr.cp.float32)
        cpu_warp = gr._warp_image_cpu(img, gr.cp.asnumpy(flow_u), gr.cp.asnumpy(flow_v), gr.cp.asnumpy(flow_w))
        gpu_warp = gr._warp_image_gpu(gr.cp.asarray(img), flow_u, flow_v, flow_w)
        np.testing.assert_allclose(cpu_warp, gr.cp.asnumpy(gpu_warp), atol=1e-5)

        # parity for 1D filtering
        filt = np.array([0.2, 0.6, 0.2], dtype=np.float32)
        cpu_f = gr._filter1_cpu(img, filt, 1)
        gpu_f = gr._filter1_gpu(gr.cp.asarray(img), gr.cp.asarray(filt), 1)
        np.testing.assert_allclose(cpu_f, gr.cp.asnumpy(gpu_f), atol=1e-6)

        # parity for boxfilter
        cpu_box = gr._boxfilter_cpu(img, 1)
        gpu_box = gr._boxfilter_gpu(gr.cp.asarray(img), 1)
        np.testing.assert_allclose(cpu_box, gr.cp.asnumpy(gpu_box), atol=1e-6)

        # parity for volumetric filtering
        cpu_vol = gr._volfilter_cpu(img, 3, 0.8)
        gpu_vol = gr._volfilter_gpu(gr.cp.asarray(img), 3, 0.8)
        np.testing.assert_allclose(cpu_vol, gr.cp.asnumpy(gpu_vol), atol=1e-6)

        # parity for 3-D interpolation helpers
        arr = np.random.rand(2, 2, 2).astype(np.float32)
        cpu_interp = gr._interp3xyz_cpu(arr, (4, 4, 4))
        gpu_interp = gr._interp3xyz_gpu(gr.cp.asarray(arr), (4, 4, 4))
        np.testing.assert_allclose(cpu_interp, gr.cp.asnumpy(gpu_interp), atol=1e-6)

        # parity for interp3xyzB helpers
        cpu_interpB = gr._interp3xyzB_cpu(arr, (4, 4, 4))
        gpu_interpB = gr._interp3xyzB_gpu(gr.cp.asarray(arr), (4, 4, 4))
        np.testing.assert_allclose(cpu_interpB, gr.cp.asnumpy(gpu_interpB), atol=1e-6)

        # parity for deformation upsampling
        u0 = np.random.rand(2, 2, 2).astype(np.float32)
        v0 = np.random.rand(2, 2, 2).astype(np.float32)
        w0 = np.random.rand(2, 2, 2).astype(np.float32)
        cu, cv, cw = gr._upsample_deformations_cpu(u0, v0, w0, (4, 4, 4))
        gu, gv, gw = gr._upsample_deformations_gpu(
            gr.cp.asarray(u0), gr.cp.asarray(v0), gr.cp.asarray(w0), (4, 4, 4)
        )
        np.testing.assert_allclose(cu, gr.cp.asnumpy(gu), atol=1e-6)
        np.testing.assert_allclose(cv, gr.cp.asnumpy(gv), atol=1e-6)
        np.testing.assert_allclose(cw, gr.cp.asnumpy(gw), atol=1e-6)

        # parity for Jacobian
        j_cpu = gr._jacobian_cpu(cu, cv, cw, 1)
        j_gpu = gr._jacobian_gpu(gu, gv, gw, 1)
        self.assertAlmostEqual(j_cpu, j_gpu, places=5)

        # parity for data cost computation
        d1 = (np.random.rand(4, 4, 4) * 1e6).astype(np.uint64)
        d2 = (np.random.rand(4, 4, 4) * 1e6).astype(np.uint64)
        cpu_dc = gr._data_cost_cpu(d1, d2, step1=1, hw=1, alpha=0.5)
        gpu_dc = gr._data_cost_gpu(gr.cp.asarray(d1), gr.cp.asarray(d2), step1=1, hw=1, alpha=0.5)
        np.testing.assert_allclose(cpu_dc, gr.cp.asnumpy(gpu_dc))

        # parity for affine warping
        X = np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=np.float32)
        cpu_aff = gr._warp_affine_cpu(img, X, gr.cp.asnumpy(flow_u), gr.cp.asnumpy(flow_v), gr.cp.asnumpy(flow_w))
        gpu_aff = gr._warp_affine_gpu(gr.cp.asarray(img), gr.cp.asarray(X), flow_u, flow_v, flow_w)
        np.testing.assert_allclose(cpu_aff, gr.cp.asnumpy(gpu_aff), atol=1e-5)

        # parity for consistent mapping
        u = np.random.rand(2, 2, 2).astype(np.float32)
        v = np.random.rand(2, 2, 2).astype(np.float32)
        w = np.random.rand(2, 2, 2).astype(np.float32)
        u2 = np.random.rand(2, 2, 2).astype(np.float32)
        v2 = np.random.rand(2, 2, 2).astype(np.float32)
        w2 = np.random.rand(2, 2, 2).astype(np.float32)
        cu1, cv1, cw1, cu2, cv2, cw2 = gr._consistent_mapping_cpu(u, v, w, u2, v2, w2, 1)
        gu1, gv1, gw1, gu2, gv2, gw2 = gr._consistent_mapping_gpu(
            gr.cp.asarray(u), gr.cp.asarray(v), gr.cp.asarray(w),
            gr.cp.asarray(u2), gr.cp.asarray(v2), gr.cp.asarray(w2), 1
        )
        np.testing.assert_allclose(cu1, gr.cp.asnumpy(gu1), atol=1e-5)
        np.testing.assert_allclose(cv1, gr.cp.asnumpy(gv1), atol=1e-5)
        np.testing.assert_allclose(cw1, gr.cp.asnumpy(gw1), atol=1e-5)
        np.testing.assert_allclose(cu2, gr.cp.asnumpy(gu2), atol=1e-5)
        np.testing.assert_allclose(cv2, gr.cp.asnumpy(gv2), atol=1e-5)
        np.testing.assert_allclose(cw2, gr.cp.asnumpy(gw2), atol=1e-5)

        # parity for warp_image_cl
        ref = np.random.rand(*img.shape).astype(np.float32)
        cpu_cl, ssd_c, ssd0_c = gr._warp_image_cl_cpu(
            img, ref, gr.cp.asnumpy(flow_u), gr.cp.asnumpy(flow_v), gr.cp.asnumpy(flow_w)
        )
        gpu_cl, ssd_g, ssd0_g = gr._warp_image_cl_gpu(
            gr.cp.asarray(img), gr.cp.asarray(ref), flow_u, flow_v, flow_w
        )
        np.testing.assert_allclose(cpu_cl, gr.cp.asnumpy(gpu_cl), atol=1e-5)
        self.assertAlmostEqual(ssd_c, ssd_g, places=5)
        self.assertAlmostEqual(ssd0_c, ssd0_g, places=5)

        # parity for warp_affine_s
        cpu_aff_s = gr._warp_affine_s_cpu(
            img, X, gr.cp.asnumpy(flow_u), gr.cp.asnumpy(flow_v), gr.cp.asnumpy(flow_w)
        )
        gpu_aff_s = gr._warp_affine_s_gpu(
            gr.cp.asarray(img), gr.cp.asarray(X), flow_u, flow_v, flow_w
        )
        np.testing.assert_allclose(cpu_aff_s, gr.cp.asnumpy(gpu_aff_s), atol=1e-5)

        # parity for messageDT
        cost = np.random.rand(3, 3, 3).astype(np.float32)
        m_cpu, ind_cpu = gr._message_dt_cpu(cost, 0.1, -0.2, 0.3)
        m_gpu, ind_gpu = gr._message_dt_gpu(gr.cp.asarray(cost), 0.1, -0.2, 0.3)
        np.testing.assert_allclose(m_cpu, gr.cp.asnumpy(m_gpu), atol=1e-6)
        np.testing.assert_array_equal(ind_cpu, gr.cp.asnumpy(ind_gpu))

        # parity for regularisation
        hw = 1
        len1 = hw * 2 + 1
        len2 = len1 ** 3
        costall = np.random.rand(2, len2).astype(np.float32)
        u0 = np.zeros(2, dtype=np.float32)
        v0 = np.zeros(2, dtype=np.float32)
        w0 = np.zeros(2, dtype=np.float32)
        ordered = np.array([0, 1], dtype=np.int32)
        parents = np.array([0, 0], dtype=np.int32)
        edgemst = np.ones(2, dtype=np.float32)
        cu1, cv1, cw1 = gr._regularisation_cpu(costall, u0, v0, w0, hw, 1, 1.0, ordered, parents, edgemst)
        gu1, gv1, gw1 = gr._regularisation_gpu(
            gr.cp.asarray(costall),
            gr.cp.asarray(u0),
            gr.cp.asarray(v0),
            gr.cp.asarray(w0),
            hw,
            1,
            1.0,
            gr.cp.asarray(ordered),
            gr.cp.asarray(parents),
            gr.cp.asarray(edgemst),
        )
        np.testing.assert_allclose(cu1, gr.cp.asnumpy(gu1), atol=1e-5)
        np.testing.assert_allclose(cv1, gr.cp.asnumpy(gv1), atol=1e-5)
        np.testing.assert_allclose(cw1, gr.cp.asnumpy(gw1), atol=1e-5)

        # parity for data cost CL
        d1 = (np.random.rand(4, 4, 4) * 1e6).astype(np.uint64)
        d2 = (np.random.rand(4, 4, 4) * 1e6).astype(np.uint64)
        cpu_dc_cl = gr._data_cost_cl_cpu(d1, d2, step1=1, hw=1, quant=1.0, alpha=0.3)
        gpu_dc_cl = gr._data_cost_cl_gpu(
            gr.cp.asarray(d1), gr.cp.asarray(d2), step1=1, hw=1, quant=1.0, alpha=0.3
        )
        np.testing.assert_allclose(cpu_dc_cl, gr.cp.asnumpy(gpu_dc_cl))

    def test_cpu_gpu_speed(self):
        from .. import gpu_registration as gr
        if not gr._gpu_ok:
            self.skipTest("GPU not available")

        shape = (512, 512, 51)
        fixed = np.zeros(shape, dtype=np.float32)
        rng = np.random.default_rng(0)
        coords = rng.integers(low=0, high=[shape[0], shape[1], shape[2]], size=(5000, 3))
        fixed[coords[:, 0], coords[:, 1], coords[:, 2]] = 1.0
        shift = (3, -2, 1)
        moving = np.roll(fixed, shift, axis=(0, 1, 2))

        for _ in range(2):
            gr._phase_corr(fixed, moving, np)
            cp_f = gr.cp.asarray(fixed)
            cp_m = gr.cp.asarray(moving)
            gr._phase_corr(cp_f, cp_m, gr.cp)
            gr.cp.cuda.runtime.deviceSynchronize()

        start = time.perf_counter()
        dy, dx, dz = gr._phase_corr(fixed, moving, np)
        cpu_out = np.roll(moving, shift=(dy, dx, dz), axis=(0, 1, 2))
        cpu_time = time.perf_counter() - start

        cp_f = gr.cp.asarray(fixed)
        cp_m = gr.cp.asarray(moving)
        gr.cp.cuda.runtime.deviceSynchronize()
        start = time.perf_counter()
        dy2, dx2, dz2 = gr._phase_corr(cp_f, cp_m, gr.cp)
        gr.cp.cuda.runtime.deviceSynchronize()
        shifted = gr.cp.roll(cp_m, shift=(dy2, dx2, dz2), axis=(0, 1, 2))
        gr.cp.cuda.runtime.deviceSynchronize()
        gpu_time = time.perf_counter() - start
        gpu_out = gr.cp.asnumpy(shifted)

        np.testing.assert_allclose(cpu_out, gpu_out, atol=1e-4)
        self.assertLess(gpu_time, cpu_time)
        print(f"CPU time: {cpu_time:.4f}s GPU time: {gpu_time:.4f}s")


if __name__ == "__main__":
    unittest.main()
