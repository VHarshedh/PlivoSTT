import unittest
import numpy as np
import pickle
from sklearn.pipeline import Pipeline

class TestEndOfTurnPipeline(unittest.TestCase):

    def test_model_loading_and_shape(self):
        """Verifies the model loads and expects the correct feature dimensions."""
        try:
            with open('model.pkl', 'rb') as f:
                model = pickle.load(f)
        except FileNotFoundError:
            self.fail("model.pkl not found. Run train_model.py first.")

        # Ensure it's the correct model type
        self.assertIsInstance(model, Pipeline, "Saved model is not a Pipeline.")
        
        # Verify the L1 feature selector successfully applied and aren't completely zeroed out
        fs = model.named_steps['fs']
        self.assertTrue(np.any(fs.estimator_.coef_ != 0), "Model coefficients are entirely zero.")

    def test_dummy_prediction(self):
        """Verifies the model can output a valid p_eot probability."""
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
            
        # 256 (Resemblyzer) + 4 (Turn duration, Language Flag, Energy Drop, Voicing Density) = 260 features
        dummy_features = np.random.rand(1, 260) 
        
        probabilities = model.predict_proba(dummy_features)
        p_eot = probabilities[0][1]
        
        self.assertGreaterEqual(p_eot, 0.0, "Probability cannot be less than 0.")
        self.assertLessEqual(p_eot, 1.0, "Probability cannot be greater than 1.")

if __name__ == '__main__':
    unittest.main()