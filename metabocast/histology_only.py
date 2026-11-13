"""
EXPERIMENTAL - morphology-only metabolite prediction.

This is a companion to the default transcriptome-based MetaboCast model, not
a replacement. It predicts the spatial metabolome from H&E histology features
alone, using the same cheap, auditable per-spot features as
add_histology_features: RGB, optical-density, texture, and tissue-fraction
summaries mapped through a ridge model to the metabolite m/z axis.

Why it exists (DESIUM and HCC histology controls of the manuscript):
  * Under leave-one-section-out on DESIUM, morphology-only prediction transferred better than
    transcriptome-only prediction, consistent with gross tissue morphology transferring across sections more
    easily than the high-dimensional transcriptome-to-metabolite map; combining both transferred best.
  * Within a section the transcriptome model is more accurate, so morphology-only prediction is a
    transfer aid, not a replacement.
  * Shuffling the H&E features collapses it to about zero, so the signal is real spatial structure.
  * On external HCC H&E, a DESIUM-trained morphology model reproduced most literature-expected
    tumour-versus-normal directions without any liver MSI supervision.
  Query features are standardised with the training mean and scale.

Strong caveats: this is a research/experimental mode. It is a coherence-level
signal, validated without measured MSI on the external HCC cohort; it uses
cheap features rather than a histology foundation model; and it is cross-tissue.
Do not treat its outputs as measured metabolites. The shipped
metabocast_model.pkl is unaffected and remains transcriptome-only.
"""

import numpy as np
from sklearn.linear_model import Ridge
from .metabocast import common_mz_axis, MZ_ATOL


class MorphologyMetabolitePredictor:
    """Experimental H&E-only metabolite predictor.

    Train on paired sections that already carry per-spot histology features
    from add_histology_features and a metabolite matrix in .uns["msi"]. Predict
    on any section with histology features, with no transcriptome or MSI needed.
    """

    def __init__(self, alpha=200.0, key="histology"):
        self.alpha = alpha
        self.key = key
        self.experimental_ = True

    def _feat(self, adata):
        if self.key not in adata.obsm:
            raise ValueError(f"adata.obsm['{self.key}'] missing - run add_histology_features(adata) first.")
        return np.asarray(adata.obsm[self.key], np.float32)

    def fit(self, adatas, metab_key="msi", mz_key="mz_features", standardize_features="global", mz_atol=MZ_ATOL):
        """Fit the morphology-only ridge.

        Parameters
        ----------
        adatas
            List of AnnData objects with .obsm[key] histology features and
            .uns[metab_key] shaped n_spots x n_metabolites.
        standardize_features
            "global" standardizes the stacked training feature matrix and
            preserves between-section contrast. "none" uses the features as is.
            Targets are per-metabolite standardized, as in MetaboCast.
        """
        if standardize_features not in {"global", "none"}:
            raise ValueError("standardize_features must be 'global' or 'none'.")
        axis = common_mz_axis(adatas, mz_key, atol=mz_atol)
        for a in adatas:
            target = np.asarray(a.uns[metab_key], float)
            if target.shape != (a.n_obs, len(axis)) or not np.isfinite(target).all() or (target < 0).any():
                raise ValueError("MSI must be finite, nonnegative and aligned to spots and ions")
        X = np.vstack([self._feat(a) for a in adatas]).astype(np.float32)
        Y = np.vstack([np.log1p(np.asarray(a.uns[metab_key], float)) for a in adatas]).astype(np.float32)
        self.feat_mode_ = standardize_features
        if standardize_features == "global":
            self.fmu_ = X.mean(0)
            self.fsd_ = X.std(0) + 1e-8
            X = (X - self.fmu_) / self.fsd_
        else:
            self.fmu_ = None
            self.fsd_ = None
        self.mz_ = axis
        self.mu_ = Y.mean(0).astype(np.float32)
        self.sd_ = (Y.std(0) + 1e-8).astype(np.float32)
        self.reg_ = Ridge(alpha=self.alpha).fit(X, (Y - self.mu_) / self.sd_)
        return self

    def predict_from_histology(self, adata, metab_uns_key="metabocast_he_pred"):
        """Predict the metabolome from histology features alone.

        Uses the training mean and scale, so predictions do not depend on which
        other spots happen to be included in the query batch.
        """
        X = self._feat(adata)
        if getattr(self, "feat_mode_", "global") == "global":
            X = (X - self.fmu_) / self.fsd_
        pred = self.reg_.predict(X.astype(np.float32)) * self.sd_ + self.mu_
        pred = np.maximum(pred, 0.0).astype(np.float32)
        adata.obsm[metab_uns_key] = pred
        return pred

    def save(self, path):
        import pickle

        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    @classmethod
    def load(cls, path):
        import pickle

        with open(path, "rb") as f:
            return pickle.load(f)
