_RELOADABLE_MODULE_NAMES = (
    "addon_info",
    "types",
    "lasso_cursor",
    "ot_keymap",
    "xraysel_ot_info",
    "geometry_tests",
    "timer",
    "view3d_utils",
    "edge_attr",
    "loop_attr",
    "poly_attr",
    "vert_attr",
    "selection_utils",
    "mesh_intersect",
    "mesh_modal",
    "mesh_ot_box",
    "mesh_ot_circle",
    "mesh_ot_lasso",
    "mesh_ot_toggle",
    "object_intersect_shared",
    "object_intersect_box",
    "object_intersect_circle",
    "object_intersect_lasso",
    "object_intersect",
    "object_modal",
    "object_ot_box",
    "object_ot_circle",
    "object_ot_lasso",
    "operators",
    "tools_keymap",
    "tools_utils",
    "tools_dummy",
    "tools_main",
    "tools",
    "keymap_ui",
    "mesh_tools_ui",
    "object_tools_ui",
    "draw",
    "keymaps_props",
    "tools_props",
    "properties",
    "addon_preferences",
    "preferences",
    "startup_handlers",
)

# Support reloading submodules
if "bpy" in locals():
    import importlib

    for module_name in _RELOADABLE_MODULE_NAMES:
        if module_name in locals():
            importlib.reload(locals()[module_name])
else:
    import bpy

    # Prevent imports when run in the background, since gpu shaders will not be available
    if not bpy.app.background:
        from . import addon_info, types  # noqa
        from .icon import lasso_cursor
        from .operators import ot_keymap, xraysel_ot_info
        from .functions import geometry_tests, timer, view3d_utils
        from .functions.mesh_attr import edge_attr, loop_attr, poly_attr, vert_attr
        from .functions.intersections import selection_utils, mesh_intersect, object_intersect
        from .functions.modals import mesh_modal
        from .operators.mesh_ot import mesh_ot_box, mesh_ot_circle, mesh_ot_lasso, mesh_ot_toggle
        from .functions.intersections.object_intersect import (
            object_intersect_shared,
            object_intersect_box,
            object_intersect_circle,
            object_intersect_lasso,
        )
        from .functions.intersections import object_intersect
        from .functions.modals import object_modal
        from .operators.object_ot import object_ot_box, object_ot_circle, object_ot_lasso
        from . import operators
        from .tools import tools_keymap, tools_utils, tools_dummy, tools_main
        from . import tools
        from .preferences.draw import keymap_ui, mesh_tools_ui, object_tools_ui
        from .preferences import draw
        from .preferences.properties import keymaps_props, tools_props
        from .preferences import addon_preferences
        from . import preferences, startup_handlers

import bpy

# Prevent loading in the background, since gpu shaders will not be available
if not bpy.app.background:
    from . import operators, preferences, startup_handlers, tools

    def register():
        preferences.register()
        operators.register()
        tools.register()
        startup_handlers.register()

    def unregister():
        preferences.unregister()
        operators.unregister()
        tools.unregister()
        startup_handlers.unregister()
else:
    register = unregister = lambda: None
