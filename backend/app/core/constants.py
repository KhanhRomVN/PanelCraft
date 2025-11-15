"""
DEPRECATED constants module.

Use app.config.constants instead of app.core.constants.

This shim re-exports all symbols from app.config.constants to preserve backward compatibility
during the refactor migration (see REFACTOR_PLAN.md). It will be removed after all imports
are updated.

TODO (cleanup phase):
- Search and replace: from app.core.constants -> from app.config import constants as C
- Remove this file when no legacy imports remain.
"""

from app.config.constants import *  # noqa: F401,F403
