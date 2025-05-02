class AssentGate:
    def __init__(self, base_threshold=1.0):
        """
        Initializes the Assent Gate.

        Args:
            base_threshold (float): The baseline decision threshold.
        """
        if base_threshold <= 0:
            raise ValueError("Base threshold must be positive.")
        self.base_threshold = base_threshold
        # Note: This gate doesn't manage collapsing bounds itself,
        # the simulation script passes the *current* threshold to check().

    def check(self, evidence_dict, current_threshold):
        """
        Checks if any action's evidence has crossed the current threshold.

        Args:
            evidence_dict (dict): Dictionary of {action_name: evidence_level}.
            current_threshold (float): The threshold value to use for this check.

        Returns:
            str or None: The name of the winning action if threshold is crossed,
                         otherwise None.
        """
        if current_threshold <= 0:
             # Avoid issues with zero/negative threshold during collapse
             current_threshold = 0.01

        winning_action = None
        max_evidence = -float('inf') # Keep track in case of simultaneous crossing

        for action, evidence in evidence_dict.items():
            # Check for positive threshold crossing
            if evidence >= current_threshold:
                # Simple rule: first past the post wins.
                # More complex rules could handle ties.
                 if evidence > max_evidence: # Prioritize higher evidence if multiple cross
                    max_evidence = evidence
                    winning_action = action

            # Optional: Check for negative threshold crossing (veto?)
            # if evidence <= -current_threshold:
            #     return f"veto_{action}" # Example veto signal

        return winning_action

# --- Test Snippet ---
if __name__ == "__main__":
    gate = AssentGate(base_threshold=1.0)
    print("Testing AssentGate...")
    evidence1 = {'A': 0.5, 'B': 0.8}
    evidence2 = {'A': 1.1, 'B': 0.9}
    evidence3 = {'A': 1.2, 'B': 1.3} # B crosses later but higher
    evidence4 = {'A': -1.1, 'B': 0.5} # Example for veto check

    print(f"Evidence: {evidence1}, Threshold: 1.0 -> Decision: {gate.check(evidence1, 1.0)}") # Should be None
    print(f"Evidence: {evidence2}, Threshold: 1.0 -> Decision: {gate.check(evidence2, 1.0)}") # Should be A
    print(f"Evidence: {evidence3}, Threshold: 1.0 -> Decision: {gate.check(evidence3, 1.0)}") # Should be B
    print(f"Evidence: {evidence3}, Threshold: 1.25 -> Decision: {gate.check(evidence3, 1.25)}") # Should be B
    print(f"Evidence: {evidence3}, Threshold: 1.4 -> Decision: {gate.check(evidence3, 1.4)}") # Should be None
    # print(f"Evidence: {evidence4}, Threshold: 1.0 -> Decision: {gate.check(evidence4, 1.0)}") # Test veto
