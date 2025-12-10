"""Agent for HTMRL environment."""


class Agent:
    """Agent for HTMRL environment."""

    def __init__(self):
        """Initialize the agent."""
        pass

    def select_action(self, state: tuple):
        """Select an action based on the current state.

        Args:
            state: The current state of the environment.

        Returns:
            The action selected by the agent.
        """
        pass

    def update_policy(self, state, action, reward, next_state):
        """Update the agent's policy based on experience.

        Args:
            experience: The experience tuple (state, action, reward, next_state).
        """
        pass
