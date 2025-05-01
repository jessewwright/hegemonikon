class RAA:
    def __init__(self, threshold=0.5, max_depth=3, decay_rate=0.8):
        """
        Recursive Adjudication Algorithm for conflict monitoring and resolution
        
        Args:
            threshold (float): Decision threshold for conflict resolution
            max_depth (int): Maximum recursion depth
            decay_rate (float): Rate at which conflict signals decay
        """
        self.threshold = threshold
        self.max_depth = max_depth
        self.decay_rate = decay_rate
        self.current_depth = 0
        self.conflict_signals = []
        self.decision_history = []

    def _monitor_conflict(self, input_signals):
        """Monitor for conflicts between input signals"""
        if len(input_signals) < 2:
            return False
            
        # Calculate conflict intensity
        signal_diff = max(input_signals) - min(input_signals)
        return signal_diff > self.threshold

    def _resolve_conflict(self, input_signals):
        """Resolve conflicts using recursive adjudication"""
        if self.current_depth >= self.max_depth:
            # Terminate recursion if max depth reached
            return self._select_majority(input_signals)
            
        # Decay conflict signals
        decayed_signals = [s * self.decay_rate for s in input_signals]
        
        # Recursively adjudicate
        self.current_depth += 1
        result = self._resolve_conflict(decayed_signals)
        self.current_depth -= 1
        
        return result

    def _select_majority(self, signals):
        """Select the majority signal when conflicts persist"""
        return max(signals)

    def update(self, input_signals):
        """
        Update the RAA with new input signals and perform adjudication
        
        Args:
            input_signals (list): List of input signals to process
            
        Returns:
            float: The resolved output signal
        """
        if not isinstance(input_signals, list):
            input_signals = [input_signals]
            
        self.conflict_signals.append(input_signals)
        
        # Check for conflicts
        has_conflict = self._monitor_conflict(input_signals)
        
        if has_conflict:
            # Resolve conflicts recursively
            output = self._resolve_conflict(input_signals)
        else:
            # No conflict, take average of signals
            output = sum(input_signals) / len(input_signals)
            
        self.decision_history.append({
            'input_signals': input_signals,
            'has_conflict': has_conflict,
            'output': output
        })
        
        return output

    def get_decision_history(self):
        """Get the history of RAA decisions"""
        return self.decision_history

if __name__ == "__main__":
    # Example usage
    raa = RAA(threshold=0.5, max_depth=3, decay_rate=0.8)
    
    # Test with conflicting signals
    print("Test with conflicting signals:")
    output = raa.update([0.9, 0.4, 0.6])
    print(f"Output: {output}")
    
    # Test with non-conflicting signals
    print("\nTest with non-conflicting signals:")
    output = raa.update([0.3, 0.35, 0.32])
    print(f"Output: {output}")
    
    # Show decision history
    print("\nDecision History:")
    for entry in raa.get_decision_history():
        print(f"Input: {entry['input_signals']}, Has Conflict: {entry['has_conflict']}, Output: {entry['output']}")
