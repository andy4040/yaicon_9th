"""Topology refinement that preserves confirmed optical material estimates."""

from .refinement import GlintRefiner


class OpticalRefiner(GlintRefiner):
    """Retain normal GLINT geometry/opacity refinement without material resets.

    The original three-set curriculum resets transmission coefficients with
    opacity. In the two-set method this would erase a confirmed optical
    hypothesis immediately after probation. Opacity retains its usual reset;
    transmission parameters and their Adam history remain learned values.
    """

    def _reset_parameter(self, name: str, value: float) -> None:
        if name == "transparency":
            return
        super()._reset_parameter(name, value)
