import bpy
import bmesh
from mathutils import Color

from . import utilities_color


class op(bpy.types.Operator):
	bl_idname = "uv.textools_color_select_vertex"
	bl_label = "Select by Vertex Color"
	bl_description = "Select faces by this vertex color"
	bl_options = {'UNDO'}
	
	index : bpy.props.IntProperty(description="Color Index", default=0)

	@classmethod
	def poll(cls, context):
		if bpy.context.area.ui_type != 'UV':
			return False
		if not bpy.context.active_object:
			return False
		if bpy.context.active_object not in bpy.context.selected_objects:
			return False
		if len(bpy.context.selected_objects) != 1:
			return False
		if bpy.context.active_object.type != 'MESH':
			return False
		return True
	

	def execute(self, context):
		select_color(self, context, self.index)
		return {'FINISHED'}



def select_color(self, context, index):
	obj = bpy.context.active_object

	if not obj.data.vertex_colors:
		self.report({'ERROR_INVALID_INPUT'}, "Object has no vertex colors")
		return
	
	if obj.mode != 'EDIT':
		bpy.ops.object.mode_set(mode='EDIT')

	bpy.ops.mesh.select_all(action='DESELECT')

	bpy.ops.object.mode_set(mode="OBJECT")
	colors = obj.data.vertex_colors.active.data
	selected_polygons = list(filter(lambda p: p.select, obj.data.polygons))

	target_color = utilities_color.get_color(index).copy()

	# op_color_assign attempts to fix the gamma. So we need to do the same here
	# so we can match what was assigned
	gamma = 2.2
	target_color[0] = pow(target_color[0],1/gamma)
	target_color[1] = pow(target_color[1],1/gamma)
	target_color[2] = pow(target_color[2],1/gamma)

	# Due the averaging color calculation we need to have a threshold.
	# Depending on the colors this might need to be adjusted by the user
	threshold = bpy.context.scene.texToolsSettings.vertex_color_threshold

	r = g = b = 0
	for p in obj.data.polygons:
		r = g = b = 0
		for i in p.loop_indices:
			c = colors[i].color
			r += c[0]
			g += c[1]
			b += c[2]
		r /= p.loop_total
		g /= p.loop_total
		b /= p.loop_total
		source_color = Color((r, g, b))

		if (abs(source_color.r - target_color.r) < threshold and
			abs(source_color.g - target_color.g) < threshold and
			abs(source_color.b - target_color.b) < threshold):

			p.select = True

	bpy.ops.object.editmode_toggle()
