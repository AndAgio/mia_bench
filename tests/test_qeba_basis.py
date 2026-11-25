"""QEBA's subspaces: the PCA basis spans the reference model's gradients, which SBA takes from its shadow model."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.helpers import IndexedDataset
from src.mia.attacks.label_only import supervised_boundary
from src.mia.attacks.label_only.helpers.qeba import block_basis, gradient_pca_basis, input_gradients
from src.mia.attacks.label_only.supervised_boundary import SupervisedBoundaryMIA
from test_hop_skip_jump import Linear, Silent


class QebaBasisTest(unittest.TestCase):
    def test_pca_basis_spans_the_directions_the_gradients_lie_in(self):
        generator = torch.Generator().manual_seed(0)
        directions = torch.linalg.qr(torch.randn(48, 2, generator=generator))[0]
        gradients = torch.randn(200, 2, generator=generator) @ directions.T
        gradients += 1e-3 * torch.randn(200, 48, generator=generator)
        basis = gradient_pca_basis(gradients, reduction_factor=2)
        self.assertEqual(tuple(basis.shape), (48, 12))
        # Each direction is (almost) entirely inside the span of the basis.
        inside = (basis.T @ directions).norm(dim=0)
        torch.testing.assert_close(inside, torch.ones(2), atol=1e-3, rtol=0)

    def test_pca_basis_keeps_one_fewer_component_than_gradients(self):
        basis = gradient_pca_basis(torch.randn(5, 48), reduction_factor=1)
        self.assertEqual(tuple(basis.shape), (48, 4))

    def test_input_gradients_of_a_linear_model_point_along_its_weights(self):
        model = Linear()
        images = 0.2 + 0.6 * torch.rand(10, 1, 4, 4, generator=torch.Generator().manual_seed(1))
        gradients = input_gradients(model, images, torch.device('cpu'), batch_size=4)
        self.assertEqual(tuple(gradients.shape), (10, 16))
        cosines = torch.nn.functional.cosine_similarity(gradients, model.w.expand(10, -1), dim=1)
        torch.testing.assert_close(cosines.abs(), torch.ones(10), atol=1e-5, rtol=0)

    def test_block_basis_has_one_indicator_per_channel_and_block(self):
        basis = block_basis((3, 32, 32), reduction_factor=2)
        self.assertEqual(tuple(basis.shape), (3072, 768))
        torch.testing.assert_close(basis.sum(dim=0), torch.full((768,), 4.))


class ShadowGradientBasisTest(unittest.TestCase):
    def test_sba_takes_gradients_on_at_most_the_cap_of_samples_outside_the_shadow_dataset(self):
        attack = object.__new__(SupervisedBoundaryMIA)
        attack.logger, attack.seed = Silent(), 0
        attack.attack_configs = SimpleNamespace(boundary=SimpleNamespace(qeba=SimpleNamespace(reduction_factor=1)))
        attack.normalization = (torch.tensor([0.5]).view(1, 1, 1, 1), torch.tensor([0.25]).view(1, 1, 1, 1))
        ids = torch.arange(20)
        images = ids.float().view(20, 1, 1, 1).expand(20, 1, 4, 4) / 100
        data = IndexedDataset(TensorDataset(images, torch.zeros(20, dtype=torch.long)), original_indices=ids.tolist())
        attack.attacker_data_distribution = SimpleNamespace(get=lambda which: data)
        outside = [3, 5, 7, 11, 13, 17]
        attack.shadow_manager = SimpleNamespace(
            shadow_data=SimpleNamespace(indices_outside_shadow_dataset=lambda index: outside),
            shadow_models=SimpleNamespace(get=lambda index: Linear()))
        seen = []

        def gradients(model, pixels, device):
            seen.extend(pixels[i] for i in range(len(pixels)))
            return torch.randn(len(pixels), 16)

        with patch.object(supervised_boundary, 'QEBA_PCA_GRADIENTS', 4), \
                patch.object(supervised_boundary, 'input_gradients', gradients):
            basis = attack.shadow_gradient_basis(device='cpu')
        self.assertEqual(len(seen), 4)
        # The images come back as pixels, so each one tells its original index.
        indices = {round(((image[0, 0, 0] - 0.5) / 0.25 * 100).item()) for image in seen}
        self.assertTrue(indices <= set(outside))
        self.assertEqual(tuple(basis.shape), (16, 3))


if __name__ == '__main__':
    unittest.main()
