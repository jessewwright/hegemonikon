import numpy as np
import math

class DiagnosticExtractor:
    def __init__(self, dt: float, early_time_threshold_ms: float = 100.0):
        """
        Initializes the DiagnosticExtractor.

        Args:
            dt (float): The time step of the DDM simulation (in seconds).
            early_time_threshold_ms (float): The time threshold for calculating
                                             early_dominance_ratio (in milliseconds).
        """
        self.dt = dt
        if self.dt <= 0:
            raise ValueError("dt must be positive.")
        
        self.feature_names_ordered = [
            'boundary_dist_upper', 'boundary_dist_lower', 'trace_variance', 
            'variance_slope', 'drift_sign_changes', 'early_dominance_ratio', 
            'work_centroid'
        ]
            
        self.early_time_threshold_steps = int(early_time_threshold_ms / (self.dt * 1000.0))
        if self.early_time_threshold_steps < 0: # Allow 0 if threshold_ms is very small or 0
            self.early_time_threshold_steps = 0

        # Initialize state variables that are reset per trial
        self.evidence_history = []
        self.drift_rates_history = []
        self.previous_variance = 0.0
        self.sign_changes = 0
        self.last_drift_sign = 0
        
        # Internal state for features calculated once or needing specific timing
        self._calculated_early_dominance_ratio = np.nan # Use np.nan for "not yet calculated"

    def reset_trial_state(self):
        """
        Resets the internal state of the extractor for a new DDM trial.
        """
        self.evidence_history = []
        self.drift_rates_history = []
        self.previous_variance = 0.0
        self.sign_changes = 0
        self.last_drift_sign = 0
        self._calculated_early_dominance_ratio = np.nan # Reset for the new trial

    def update_and_extract_features(self, current_evidence: float, 
                                    accumulated_ddm_time: float, 
                                    current_drift_rate: float, 
                                    upper_boundary: float, 
                                    lower_boundary: float):
        """
        Updates internal state with data from the current DDM step and calculates
        real-time diagnostic features.
        """
        self.evidence_history.append(current_evidence)
        self.drift_rates_history.append(current_drift_rate)

        current_sign = np.sign(current_drift_rate)
        if self.last_drift_sign != 0 and current_sign != 0 and current_sign != self.last_drift_sign:
            self.sign_changes += 1
        if current_sign != 0:
            self.last_drift_sign = current_sign

        dist_to_upper = upper_boundary - current_evidence
        dist_to_lower = current_evidence - lower_boundary 

        current_variance = 0.0
        if len(self.evidence_history) > 1:
            current_variance = np.var(self.evidence_history)
        
        variance_slope = 0.0
        if self.dt > 0 and len(self.evidence_history) > 1:
             variance_slope = (current_variance - self.previous_variance) / self.dt
        self.previous_variance = current_variance

        current_step_index = len(self.evidence_history) - 1
        if current_step_index == self.early_time_threshold_steps:
            if upper_boundary != 0:
                 self._calculated_early_dominance_ratio = np.abs(current_evidence) / upper_boundary
            else:
                 self._calculated_early_dominance_ratio = np.nan

        centroid = 0.0
        if self.evidence_history:
            times = np.arange(len(self.evidence_history)) * self.dt
            abs_evidence_trace = np.abs(self.evidence_history)
            sum_abs_evidence = np.sum(abs_evidence_trace)
            if sum_abs_evidence > 1e-9: 
                weighted_sum = np.sum(abs_evidence_trace * times)
                centroid = weighted_sum / sum_abs_evidence
            elif len(self.evidence_history) > 0: 
                centroid = (len(self.evidence_history) -1) * self.dt / 2.0
            
        return {
            'boundary_dist_upper': dist_to_upper,
            'boundary_dist_lower': dist_to_lower,
            'trace_variance': current_variance,
            'variance_slope': variance_slope,
            'drift_sign_changes': self.sign_changes,
            'early_dominance_ratio': self._calculated_early_dominance_ratio,
            'work_centroid': centroid
        }

class MetaCognitiveClassifier:
    def __init__(self, feature_names: list[str], num_classes: int = 3):
        """
        Initializes the MetaCognitiveClassifier.

        Args:
            feature_names (list[str]): Ordered list of feature names.
            num_classes (int): Number of metacognitive classes.
        """
        self.feature_names = feature_names
        self.num_classes = num_classes
        self.class_labels = ['stable_adherence', 'override_in_progress', 'unresolved_conflict']
        if len(self.class_labels) != num_classes:
            raise ValueError(f"Number of class_labels must match num_classes ({num_classes}).")

        # Features: boundary_dist_upper, boundary_dist_lower, trace_variance, 
        #           variance_slope, drift_sign_changes, early_dominance_ratio, work_centroid
        # Expected order based on self.feature_names
        if len(self.feature_names) != 7:
            raise ValueError(f"Expected 7 feature names, got {len(self.feature_names)}")

        self.weights = np.array([
            # stable_adherence
            [0.1,  0.1, -0.5, -0.3, -0.4,  0.8,  0.1],
            # override_in_progress
            [-0.2, 0.5,  0.3,  0.4,  0.1, -0.2,  0.2],
            # unresolved_conflict
            [0.0,  0.0,  0.6,  0.1,  0.5, -0.5, -0.2]
        ])
        self.biases = np.array([0.0, 0.0, 0.0]) # Shape (num_classes,)

        if self.weights.shape != (self.num_classes, len(self.feature_names)):
            raise ValueError(f"Weights shape mismatch. Expected {(self.num_classes, len(self.feature_names))}, got {self.weights.shape}")
        if self.biases.shape != (self.num_classes,):
            raise ValueError(f"Biases shape mismatch. Expected {(self.num_classes,)}, got {self.biases.shape}")


    def classify(self, features_dict: dict) -> dict[str, float]:
        """
        Classifies the current DDM state based on extracted features.

        Args:
            features_dict (dict): A dictionary of feature names and their values.

        Returns:
            dict: A dictionary mapping class labels to their probabilities.
        """
        feature_vector = np.zeros(len(self.feature_names))
        for i, name in enumerate(self.feature_names):
            value = features_dict.get(name, 0.0) # Default to 0.0 if feature is missing
            if name == 'early_dominance_ratio' and np.isnan(value):
                value = 0.0 # Replace NaN EDR with 0.0 for dot product
            feature_vector[i] = value
        
        logits = np.dot(self.weights, feature_vector) + self.biases
        
        # Softmax for probabilities
        exp_logits = np.exp(logits - np.max(logits)) # Subtract max for numerical stability
        probabilities = exp_logits / np.sum(exp_logits)
        
        return {self.class_labels[i]: probabilities[i] for i in range(self.num_classes)}


if __name__ == '__main__':
    # Example Usage for DiagnosticExtractor
    dt_sim = 0.001  # 1 ms
    extractor = DiagnosticExtractor(dt=dt_sim, early_time_threshold_ms=50) 
    print(f"Early dominance ratio will be calculated at step: {extractor.early_time_threshold_steps}")

    test_upper_boundary = 1.0
    test_lower_boundary = -1.0
    
    print("\n--- Trial 1 (DiagnosticExtractor) ---")
    extractor.reset_trial_state()
    mock_evidence_trace = [0.0, 0.1, 0.05, 0.15, 0.25, 0.3, 0.2, 0.1, 0.15, 0.25] 
    mock_drift_rates = [1.0, -0.5, 1.0, 1.0, 0.5, -1.0, -1.0, 0.5, 1.0, 1.0]
    
    last_features_trial1 = {}
    for i in range(len(mock_evidence_trace)):
        features = extractor.update_and_extract_features(
            current_evidence=mock_evidence_trace[i],
            accumulated_ddm_time=i * dt_sim,
            current_drift_rate=mock_drift_rates[i],
            upper_boundary=test_upper_boundary,
            lower_boundary=test_lower_boundary
        )
        last_features_trial1 = features # Store the last set of features for the classifier
        if i < 3 or i == extractor.early_time_threshold_steps or i == len(mock_evidence_trace) -1 :
            print(f"Step {i}: Features = {features}")

    print("\n--- Trial 2 (DiagnosticExtractor after reset) ---")
    extractor.reset_trial_state()
    mock_evidence_trace_2 = [-0.05, -0.15, -0.1, -0.2]
    mock_drift_rates_2 = [-0.5, -1.0, 0.5, -1.0]
    for i in range(len(mock_evidence_trace_2)):
        features = extractor.update_and_extract_features(
            current_evidence=mock_evidence_trace_2[i],
            accumulated_ddm_time=i * dt_sim,
            current_drift_rate=mock_drift_rates_2[i],
            upper_boundary=test_upper_boundary,
            lower_boundary=test_lower_boundary
        )
        print(f"Step {i}: Features = {features}")
    
    print("\n--- Trial 3 (DiagnosticExtractor - testing early_dominance_ratio exact timing) ---")
    extractor.reset_trial_state() 
    print(f"Target step for EDR: {extractor.early_time_threshold_steps}")
    edr_value_trial3 = np.nan
    for i in range(extractor.early_time_threshold_steps + 5): 
        ev = i * 0.01 
        dr = 0.1 if i % 2 == 0 else -0.1 
        features = extractor.update_and_extract_features(
            current_evidence=ev,
            accumulated_ddm_time=i * dt_sim,
            current_drift_rate=dr,
            upper_boundary=test_upper_boundary,
            lower_boundary=test_lower_boundary
        )
        if i == extractor.early_time_threshold_steps:
            print(f"Step {i} (EDR calculation step): Features = {features}")
            edr_value_trial3 = features['early_dominance_ratio']
        elif i == extractor.early_time_threshold_steps + 1:
             print(f"Step {i} (after EDR calculation): Features = {features}")
             assert features['early_dominance_ratio'] == edr_value_trial3, "EDR should persist after calculation"

    print(f"Final EDR from trial 3: {edr_value_trial3}")
    assert not np.isnan(edr_value_trial3), "EDR should have been calculated."

    # Example Usage for MetaCognitiveClassifier
    print("\n--- MetaCognitiveClassifier Test ---")
    feature_order = ['boundary_dist_upper', 'boundary_dist_lower', 'trace_variance', 
                     'variance_slope', 'drift_sign_changes', 'early_dominance_ratio', 
                     'work_centroid']
    
    classifier = MetaCognitiveClassifier(feature_names=feature_order)

    # Using features from the end of Trial 1 of DiagnosticExtractor
    print(f"Features used for classification (from end of Trial 1): {last_features_trial1}")
    classification_probs = classifier.classify(last_features_trial1)
    print(f"Classification Probabilities: {classification_probs}")

    # Example with a feature dictionary that might be missing a key or have NaN EDR
    sample_features_custom = {
        'boundary_dist_upper': 0.7,
        'boundary_dist_lower': 0.3, # Lower is closer
        'trace_variance': 0.05,
        'variance_slope': 0.2, # Increasing variance
        'drift_sign_changes': 2,
        'early_dominance_ratio': np.nan, # EDR not yet computed
        'work_centroid': 0.003
        # 'some_other_feature': 123 # This would be ignored
    }
    print(f"\nCustom Sample Features for classification: {sample_features_custom}")
    classification_probs_custom = classifier.classify(sample_features_custom)
    print(f"Classification Probabilities (Custom): {classification_probs_custom}")
```
