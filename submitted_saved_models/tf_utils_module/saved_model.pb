¦·
Ζͺ
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
Ύ
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring "serve*2.5.02unknown8οΝ

NoOpNoOp
i
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*%
valueB B


signatures
 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCallStatefulPartitionedCallsaver_filenameConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *'
f"R 
__inference__traced_save_62145

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 **
f%R#
!__inference__traced_restore_62155έΗ
’

(__inference_crop_and_pad_with_bbox_61446	
image
spacing

bbox_start
bbox_end
identity

identity_1

identity_2C
ShapeShapeimage*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2Ξ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_sliceW
truedivRealDiv
bbox_startspacing*
T0*
_output_shapes
:2	
truedivI
FloorFloortruediv:z:0*
T0*
_output_shapes
:2
FloorS
CastCast	Floor:y:0*

DstT0*

SrcT0*
_output_shapes
:2
CastX
	Maximum/yConst*
_output_shapes
: *
dtype0*
value	B : 2
	Maximum/y`
MaximumMaximumCast:y:0Maximum/y:output:0*
T0*
_output_shapes
:2	
MaximumM
subSubMaximum:z:0Cast:y:0*
T0*
_output_shapes
:2
subS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
add/yR
addAddV2bbox_endadd/y:output:0*
T0*
_output_shapes
:2
addX
	truediv_1RealDivadd:z:0spacing*
T0*
_output_shapes
:2
	truediv_1H
CeilCeiltruediv_1:z:0*
T0*
_output_shapes
:2
CeilV
Cast_1CastCeil:y:0*

DstT0*

SrcT0*
_output_shapes
:2
Cast_1f
MinimumMinimum
Cast_1:y:0strided_slice:output:0*
T0*
_output_shapes
:2	
MinimumS
sub_1Sub
Cast_1:y:0Minimum:z:0*
T0*
_output_shapes
:2
sub_1h
stackPacksub:z:0	sub_1:z:0*
N*
T0*
_output_shapes

:*

axis2
stack_
sub_2Substrided_slice:output:0Minimum:z:0*
T0*
_output_shapes
:2
sub_2p
stack_1PackMaximum:z:0	sub_2:z:0*
N*
T0*
_output_shapes

:*

axis2	
stack_1Ψ
PartitionedCallPartitionedCallimagestack_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8????????????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *%
f R
__inference_crop_image_614312
PartitionedCallυ
PartitionedCall_1PartitionedCallPartitionedCall:output:0stack:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:A?????????????????????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *$
fR
__inference_pad_image_614412
PartitionedCall_1
IdentityIdentityPartitionedCall_1:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2

Identity]

Identity_1Identitystack:output:0*
T0*
_output_shapes

:2

Identity_1_

Identity_2Identitystack_1:output:0*
T0*
_output_shapes

:2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L:8????????????????????????????????????::::u q
N
_output_shapes<
::8????????????????????????????????????

_user_specified_nameimage:C?

_output_shapes
:
!
_user_specified_name	spacing:FB

_output_shapes
:
$
_user_specified_name
bbox_start:D@

_output_shapes
:
"
_user_specified_name
bbox_end

Z
"__inference_size_for_spacing_61149
size
spacing
new_spacing
identityN
CastCastsize*

DstT0*

SrcT0*
_output_shapes
:2
CastI
mulMulCast:y:0spacing*
T0*
_output_shapes
:2
mulX
truedivRealDivmul:z:0new_spacing*
T0*
_output_shapes
:2	
truedivF
CeilCeiltruediv:z:0*
T0*
_output_shapes
:2
CeilV
Cast_1CastCeil:y:0*

DstT0*

SrcT0*
_output_shapes
:2
Cast_1Q
IdentityIdentity
Cast_1:y:0*
T0*
_output_shapes
:2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
::::@ <

_output_shapes
:

_user_specified_namesize:C?

_output_shapes
:
!
_user_specified_name	spacing:GC

_output_shapes
:
%
_user_specified_namenew_spacing
ύ
f
cond_false_60655
cond_placeholder
cond_shape_image

cond_identity
cond_identity_1Z

cond/ConstConst*
_output_shapes
: *
dtype0*
value	B : 2

cond/ConstX

cond/ShapeShapecond_shape_image*
T0
*
_output_shapes
:2

cond/Shape~
cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
cond/strided_slice/stack
cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
cond/strided_slice/stack_1
cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
cond/strided_slice/stack_2
cond/strided_sliceStridedSlicecond/Shape:output:0!cond/strided_slice/stack:output:0#cond/strided_slice/stack_1:output:0#cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
cond/strided_sliceZ

cond/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2

cond/sub/yn
cond/subSubcond/strided_slice:output:0cond/sub/y:output:0*
T0*
_output_shapes
: 2

cond/subY
cond/IdentityIdentitycond/sub:z:0*
T0*
_output_shapes
: 2
cond/Identityd
cond/Identity_1Identitycond/Const:output:0*
T0*
_output_shapes
: 2
cond/Identity_1"'
cond_identitycond/Identity:output:0"+
cond_identity_1cond/Identity_1:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:?????????:'???????????????????????????:- )
'
_output_shapes
:?????????:C?
=
_output_shapes+
):'???????????????????????????
C
F
__inference_crop_image_61797	
image
	croppings
identityC
ShapeShapeimage*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2Ξ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_1
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2η
strided_slice_1StridedSlice	croppingsstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2τ
strided_slice_2StridedSlicestrided_slice:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_1
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2η
strided_slice_3StridedSlice	croppingsstrided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3f
subSubstrided_slice_2:output:0strided_slice_3:output:0*
T0*
_output_shapes
: 2
sub
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stack
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_4/stack_1
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_4/stack_2η
strided_slice_4StridedSlice	croppingsstrided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_4x
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_5/stack|
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_5/stack_1|
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_5/stack_2τ
strided_slice_5StridedSlicestrided_slice:output:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_5
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_6/stack
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_6/stack_1
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_6/stack_2η
strided_slice_6StridedSlice	croppingsstrided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_6j
sub_1Substrided_slice_5:output:0strided_slice_6:output:0*
T0*
_output_shapes
: 2
sub_1
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_7/stack
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_7/stack_1
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_7/stack_2η
strided_slice_7StridedSlice	croppingsstrided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_7x
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_8/stack|
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_8/stack_1|
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_8/stack_2τ
strided_slice_8StridedSlicestrided_slice:output:0strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_8
strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_9/stack
strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_9/stack_1
strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_9/stack_2η
strided_slice_9StridedSlice	croppingsstrided_slice_9/stack:output:0 strided_slice_9/stack_1:output:0 strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_9j
sub_2Substrided_slice_8:output:0strided_slice_9:output:0*
T0*
_output_shapes
: 2
sub_2v
strided_slice_10/stack/0Const*
_output_shapes
: *
dtype0*
value	B : 2
strided_slice_10/stack/0v
strided_slice_10/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 2
strided_slice_10/stack/1ϊ
strided_slice_10/stackPack!strided_slice_10/stack/0:output:0!strided_slice_10/stack/1:output:0strided_slice_1:output:0strided_slice_4:output:0strided_slice_7:output:0*
N*
T0*
_output_shapes
:2
strided_slice_10/stackz
strided_slice_10/stack_1/0Const*
_output_shapes
: *
dtype0*
value	B : 2
strided_slice_10/stack_1/0z
strided_slice_10/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : 2
strided_slice_10/stack_1/1Σ
strided_slice_10/stack_1Pack#strided_slice_10/stack_1/0:output:0#strided_slice_10/stack_1/1:output:0sub:z:0	sub_1:z:0	sub_2:z:0*
N*
T0*
_output_shapes
:2
strided_slice_10/stack_1
strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*)
value B"               2
strided_slice_10/stack_2ͺ
strided_slice_10StridedSliceimagestrided_slice_10/stack:output:0!strided_slice_10/stack_1:output:0!strided_slice_10/stack_2:output:0*
Index0*
T0*N
_output_shapes<
::8????????????????????????????????????*

begin_mask*
end_mask2
strided_slice_10
IdentityIdentitystrided_slice_10:output:0*
T0*N
_output_shapes<
::8????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:8????????????????????????????????????::u q
N
_output_shapes<
::8????????????????????????????????????

_user_specified_nameimage:IE

_output_shapes

:
#
_user_specified_name	croppings
Κ
p
cond_1_false_61585
cond_1_placeholder
cond_1_shape_image

cond_1_identity
cond_1_identity_1^
cond_1/ConstConst*
_output_shapes
: *
dtype0*
value	B : 2
cond_1/Const^
cond_1/ShapeShapecond_1_shape_image*
T0
*
_output_shapes
:2
cond_1/Shape
cond_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
cond_1/strided_slice/stack
cond_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
cond_1/strided_slice/stack_1
cond_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
cond_1/strided_slice/stack_2
cond_1/strided_sliceStridedSlicecond_1/Shape:output:0#cond_1/strided_slice/stack:output:0%cond_1/strided_slice/stack_1:output:0%cond_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
cond_1/strided_slice^
cond_1/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2
cond_1/sub/yv

cond_1/subSubcond_1/strided_slice:output:0cond_1/sub/y:output:0*
T0*
_output_shapes
: 2

cond_1/sub_
cond_1/IdentityIdentitycond_1/sub:z:0*
T0*
_output_shapes
: 2
cond_1/Identityj
cond_1/Identity_1Identitycond_1/Const:output:0*
T0*
_output_shapes
: 2
cond_1/Identity_1"+
cond_1_identitycond_1/Identity:output:0"/
cond_1_identity_1cond_1/Identity_1:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:?????????:'???????????????????????????:- )
'
_output_shapes
:?????????:C?
=
_output_shapes+
):'???????????????????????????
Z
G
__inference_resize_image_62022	
image
new_size
identityU
resize3d/ShapeShapeimage*
T0*
_output_shapes
:2
resize3d/Shape
resize3d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
resize3d/strided_slice/stack
resize3d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
resize3d/strided_slice/stack_1
resize3d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
resize3d/strided_slice/stack_2
resize3d/strided_sliceStridedSliceresize3d/Shape:output:0%resize3d/strided_slice/stack:output:0'resize3d/strided_slice/stack_1:output:0'resize3d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
resize3d/strided_slice
resize3d/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2 
resize3d/strided_slice_1/stack
 resize3d/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_1/stack_1
 resize3d/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_1/stack_2’
resize3d/strided_slice_1StridedSliceresize3d/Shape:output:0'resize3d/strided_slice_1/stack:output:0)resize3d/strided_slice_1/stack_1:output:0)resize3d/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
resize3d/strided_slice_1
resize3d/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2 
resize3d/strided_slice_2/stack
 resize3d/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_2/stack_1
 resize3d/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_2/stack_2
resize3d/strided_slice_2StridedSliceresize3d/Shape:output:0'resize3d/strided_slice_2/stack:output:0)resize3d/strided_slice_2/stack_1:output:0)resize3d/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
resize3d/strided_slice_2
resize3d/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
resize3d/strided_slice_3/stack
 resize3d/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_3/stack_1
 resize3d/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_3/stack_2¬
resize3d/strided_slice_3StridedSlice!resize3d/strided_slice_2:output:0'resize3d/strided_slice_3/stack:output:0)resize3d/strided_slice_3/stack_1:output:0)resize3d/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
resize3d/strided_slice_3
resize3d/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:2 
resize3d/strided_slice_4/stack
 resize3d/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_4/stack_1
 resize3d/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_4/stack_2¬
resize3d/strided_slice_4StridedSlice!resize3d/strided_slice_2:output:0'resize3d/strided_slice_4/stack:output:0)resize3d/strided_slice_4/stack_1:output:0)resize3d/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
resize3d/strided_slice_4
resize3d/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:2 
resize3d/strided_slice_5/stack
 resize3d/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_5/stack_1
 resize3d/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_5/stack_2¬
resize3d/strided_slice_5StridedSlice!resize3d/strided_slice_2:output:0'resize3d/strided_slice_5/stack:output:0)resize3d/strided_slice_5/stack_1:output:0)resize3d/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
resize3d/strided_slice_5
resize3d/transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                2
resize3d/transpose/permΐ
resize3d/transpose	Transposeimage resize3d/transpose/perm:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2
resize3d/transpose
resize3d/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
resize3d/strided_slice_6/stack
 resize3d/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_6/stack_1
 resize3d/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_6/stack_2
resize3d/strided_slice_6StridedSlicenew_size'resize3d/strided_slice_6/stack:output:0)resize3d/strided_slice_6/stack_1:output:0)resize3d/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
resize3d/strided_slice_6
resize3d/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB:2 
resize3d/strided_slice_7/stack
 resize3d/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_7/stack_1
 resize3d/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_7/stack_2
resize3d/strided_slice_7StridedSlicenew_size'resize3d/strided_slice_7/stack:output:0)resize3d/strided_slice_7/stack_1:output:0)resize3d/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
resize3d/strided_slice_7
resize3d/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB:2 
resize3d/strided_slice_8/stack
 resize3d/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_8/stack_1
 resize3d/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_8/stack_2
resize3d/strided_slice_8StridedSlicenew_size'resize3d/strided_slice_8/stack:output:0)resize3d/strided_slice_8/stack_1:output:0)resize3d/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
resize3d/strided_slice_8
resize3d/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
resize3d/Reshape/shape/0ς
resize3d/Reshape/shapePack!resize3d/Reshape/shape/0:output:0!resize3d/strided_slice_4:output:0!resize3d/strided_slice_5:output:0!resize3d/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
resize3d/Reshape/shape½
resize3d/ReshapeReshaperesize3d/transpose:y:0resize3d/Reshape/shape:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
resize3d/Reshape¨
resize3d/resize/sizePack!resize3d/strided_slice_7:output:0!resize3d/strided_slice_8:output:0*
N*
T0*
_output_shapes
:2
resize3d/resize/sizeΥ
resize3d/resize/ResizeArea
ResizeArearesize3d/Reshape:output:0resize3d/resize/size:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
resize3d/resize/ResizeArea·
resize3d/CastCast+resize3d/resize/ResizeArea:resized_images:0*

DstT0*

SrcT0*J
_output_shapes8
6:4????????????????????????????????????2
resize3d/Cast
resize3d/Reshape_1/shapePackresize3d/strided_slice:output:0!resize3d/strided_slice_3:output:0!resize3d/strided_slice_7:output:0!resize3d/strided_slice_8:output:0!resize3d/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
resize3d/Reshape_1/shapeΛ
resize3d/Reshape_1Reshaperesize3d/Cast:y:0!resize3d/Reshape_1/shape:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2
resize3d/Reshape_1
resize3d/transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                2
resize3d/transpose_1/permά
resize3d/transpose_1	Transposeresize3d/Reshape_1:output:0"resize3d/transpose_1/perm:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2
resize3d/transpose_1
resize3d/Reshape_2/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
resize3d/Reshape_2/shape/0ψ
resize3d/Reshape_2/shapePack#resize3d/Reshape_2/shape/0:output:0!resize3d/strided_slice_7:output:0!resize3d/strided_slice_3:output:0!resize3d/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
resize3d/Reshape_2/shapeΕ
resize3d/Reshape_2Reshaperesize3d/transpose_1:y:0!resize3d/Reshape_2/shape:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
resize3d/Reshape_2¬
resize3d/resize_1/sizePack!resize3d/strided_slice_7:output:0!resize3d/strided_slice_6:output:0*
N*
T0*
_output_shapes
:2
resize3d/resize_1/sizeέ
resize3d/resize_1/ResizeArea
ResizeArearesize3d/Reshape_2:output:0resize3d/resize_1/size:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
resize3d/resize_1/ResizeArea½
resize3d/Cast_1Cast-resize3d/resize_1/ResizeArea:resized_images:0*

DstT0*

SrcT0*J
_output_shapes8
6:4????????????????????????????????????2
resize3d/Cast_1
resize3d/Reshape_3/shapePackresize3d/strided_slice:output:0!resize3d/strided_slice_8:output:0!resize3d/strided_slice_7:output:0!resize3d/strided_slice_6:output:0!resize3d/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
resize3d/Reshape_3/shapeΝ
resize3d/Reshape_3Reshaperesize3d/Cast_1:y:0!resize3d/Reshape_3/shape:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2
resize3d/Reshape_3
resize3d/transpose_2/permConst*
_output_shapes
:*
dtype0*)
value B"                2
resize3d/transpose_2/permά
resize3d/transpose_2	Transposeresize3d/Reshape_3:output:0"resize3d/transpose_2/perm:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2
resize3d/transpose_2
IdentityIdentityresize3d/transpose_2:y:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:A?????????????????????????????????????????????::~ z
W
_output_shapesE
C:A?????????????????????????????????????????????

_user_specified_nameimage:D@

_output_shapes
:
"
_user_specified_name
new_size
η	
D
__inference_pad_image_61876	
image
paddings
identity
concat/values_0Const*
_output_shapes

:*
dtype0*)
value B"                2
concat/values_0\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis
concatConcatV2concat/values_0:output:0paddingsconcat/axis:output:0*
N*
T0*
_output_shapes

:2
concatr
PadV2/constant_valuesConst*
_output_shapes
: *
dtype0*
valueB	 jΘ2
PadV2/constant_values±
PadV2PadV2imageconcat:output:0PadV2/constant_values:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2
PadV2
IdentityIdentityPadV2:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:A?????????????????????????????????????????????::~ z
W
_output_shapesE
C:A?????????????????????????????????????????????

_user_specified_nameimage:HD

_output_shapes

:
"
_user_specified_name
paddings
C
F
__inference_crop_image_61431	
image
	croppings
identityC
ShapeShapeimage*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2Ξ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_1
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2η
strided_slice_1StridedSlice	croppingsstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2τ
strided_slice_2StridedSlicestrided_slice:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_1
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2η
strided_slice_3StridedSlice	croppingsstrided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3f
subSubstrided_slice_2:output:0strided_slice_3:output:0*
T0*
_output_shapes
: 2
sub
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stack
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_4/stack_1
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_4/stack_2η
strided_slice_4StridedSlice	croppingsstrided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_4x
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_5/stack|
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_5/stack_1|
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_5/stack_2τ
strided_slice_5StridedSlicestrided_slice:output:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_5
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_6/stack
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_6/stack_1
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_6/stack_2η
strided_slice_6StridedSlice	croppingsstrided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_6j
sub_1Substrided_slice_5:output:0strided_slice_6:output:0*
T0*
_output_shapes
: 2
sub_1
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_7/stack
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_7/stack_1
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_7/stack_2η
strided_slice_7StridedSlice	croppingsstrided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_7x
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_8/stack|
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_8/stack_1|
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_8/stack_2τ
strided_slice_8StridedSlicestrided_slice:output:0strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_8
strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_9/stack
strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_9/stack_1
strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_9/stack_2η
strided_slice_9StridedSlice	croppingsstrided_slice_9/stack:output:0 strided_slice_9/stack_1:output:0 strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_9j
sub_2Substrided_slice_8:output:0strided_slice_9:output:0*
T0*
_output_shapes
: 2
sub_2v
strided_slice_10/stack/0Const*
_output_shapes
: *
dtype0*
value	B : 2
strided_slice_10/stack/0v
strided_slice_10/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 2
strided_slice_10/stack/1ϊ
strided_slice_10/stackPack!strided_slice_10/stack/0:output:0!strided_slice_10/stack/1:output:0strided_slice_1:output:0strided_slice_4:output:0strided_slice_7:output:0*
N*
T0*
_output_shapes
:2
strided_slice_10/stackz
strided_slice_10/stack_1/0Const*
_output_shapes
: *
dtype0*
value	B : 2
strided_slice_10/stack_1/0z
strided_slice_10/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : 2
strided_slice_10/stack_1/1Σ
strided_slice_10/stack_1Pack#strided_slice_10/stack_1/0:output:0#strided_slice_10/stack_1/1:output:0sub:z:0	sub_1:z:0	sub_2:z:0*
N*
T0*
_output_shapes
:2
strided_slice_10/stack_1
strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*)
value B"               2
strided_slice_10/stack_2ͺ
strided_slice_10StridedSliceimagestrided_slice_10/stack:output:0!strided_slice_10/stack_1:output:0!strided_slice_10/stack_2:output:0*
Index0*
T0*N
_output_shapes<
::8????????????????????????????????????*

begin_mask*
end_mask2
strided_slice_10
IdentityIdentitystrided_slice_10:output:0*
T0*N
_output_shapes<
::8????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:8????????????????????????????????????::u q
N
_output_shapes<
::8????????????????????????????????????

_user_specified_nameimage:IE

_output_shapes

:
#
_user_specified_name	croppings
Ή
D
__inference_pad_image_61885	
image
paddings
identity
concat/values_0Const*
_output_shapes

:*
dtype0*)
value B"                2
concat/values_0\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis
concatConcatV2concat/values_0:output:0paddingsconcat/axis:output:0*
N*
T0*
_output_shapes

:2
concat
PadPadimageconcat:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2
Pad
IdentityIdentityPad:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:8????????????????????????????????????::u q
N
_output_shapes<
::8????????????????????????????????????

_user_specified_nameimage:HD

_output_shapes

:
"
_user_specified_name
paddings
Ε
x
cond_2_true_61628
cond_2_strided_slice_cast_2
cond_2_placeholder

cond_2_identity
cond_2_identity_1
cond_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
cond_2/strided_slice/stack
cond_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      2
cond_2/strided_slice/stack_1
cond_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
cond_2/strided_slice/stack_2
cond_2/strided_sliceStridedSlicecond_2_strided_slice_cast_2#cond_2/strided_slice/stack:output:0%cond_2/strided_slice/stack_1:output:0%cond_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
cond_2/strided_slice
cond_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"????    2
cond_2/strided_slice_1/stack
cond_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2 
cond_2/strided_slice_1/stack_1
cond_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2 
cond_2/strided_slice_1/stack_2
cond_2/strided_slice_1StridedSlicecond_2_strided_slice_cast_2%cond_2/strided_slice_1/stack:output:0'cond_2/strided_slice_1/stack_1:output:0'cond_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
cond_2/strided_slice_1p
cond_2/IdentityIdentitycond_2/strided_slice_1:output:0*
T0*
_output_shapes
: 2
cond_2/Identityr
cond_2/Identity_1Identitycond_2/strided_slice:output:0*
T0*
_output_shapes
: 2
cond_2/Identity_1"+
cond_2_identitycond_2/Identity:output:0"/
cond_2_identity_1cond_2/Identity_1:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:?????????:'???????????????????????????:- )
'
_output_shapes
:?????????:C?
=
_output_shapes+
):'???????????????????????????
’

(__inference_crop_and_pad_with_bbox_61740	
image
spacing

bbox_start
bbox_end
identity

identity_1

identity_2C
ShapeShapeimage*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2Ξ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_sliceW
truedivRealDiv
bbox_startspacing*
T0*
_output_shapes
:2	
truedivI
FloorFloortruediv:z:0*
T0*
_output_shapes
:2
FloorS
CastCast	Floor:y:0*

DstT0*

SrcT0*
_output_shapes
:2
CastX
	Maximum/yConst*
_output_shapes
: *
dtype0*
value	B : 2
	Maximum/y`
MaximumMaximumCast:y:0Maximum/y:output:0*
T0*
_output_shapes
:2	
MaximumM
subSubMaximum:z:0Cast:y:0*
T0*
_output_shapes
:2
subS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
add/yR
addAddV2bbox_endadd/y:output:0*
T0*
_output_shapes
:2
addX
	truediv_1RealDivadd:z:0spacing*
T0*
_output_shapes
:2
	truediv_1H
CeilCeiltruediv_1:z:0*
T0*
_output_shapes
:2
CeilV
Cast_1CastCeil:y:0*

DstT0*

SrcT0*
_output_shapes
:2
Cast_1f
MinimumMinimum
Cast_1:y:0strided_slice:output:0*
T0*
_output_shapes
:2	
MinimumS
sub_1Sub
Cast_1:y:0Minimum:z:0*
T0*
_output_shapes
:2
sub_1h
stackPacksub:z:0	sub_1:z:0*
N*
T0*
_output_shapes

:*

axis2
stack_
sub_2Substrided_slice:output:0Minimum:z:0*
T0*
_output_shapes
:2
sub_2p
stack_1PackMaximum:z:0	sub_2:z:0*
N*
T0*
_output_shapes

:*

axis2	
stack_1Ψ
PartitionedCallPartitionedCallimagestack_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8????????????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *%
f R
__inference_crop_image_614312
PartitionedCallυ
PartitionedCall_1PartitionedCallPartitionedCall:output:0stack:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:A?????????????????????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *$
fR
__inference_pad_image_614412
PartitionedCall_1
IdentityIdentityPartitionedCall_1:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2

Identity]

Identity_1Identitystack:output:0*
T0*
_output_shapes

:2

Identity_1_

Identity_2Identitystack_1:output:0*
T0*
_output_shapes

:2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L:8????????????????????????????????????::::u q
N
_output_shapes<
::8????????????????????????????????????

_user_specified_nameimage:C?

_output_shapes
:
!
_user_specified_name	spacing:FB

_output_shapes
:
$
_user_specified_name
bbox_start:D@

_output_shapes
:
"
_user_specified_name
bbox_end
¨ 
u
/__inference_bbox_from_localization_output_60803	
image

spacing
paddings
identity

identity_1
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"          2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice/stack_2
strided_sliceStridedSliceimagestrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0
*=
_output_shapes+
):'???????????????????????????*
ellipsis_mask*
shrink_axis_mask2
strided_slice«
PartitionedCallPartitionedCallstrided_slice:output:0*
Tin
2
*
Tout
2*
_collective_manager_ids
 * 
_output_shapes
::* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *'
f"R 
__inference_bounding_box_607782
PartitionedCall
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack_1
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2
strided_slice_1StridedSlicepaddingsstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_1j
subSubPartitionedCall:output:0strided_slice_1:output:0*
T0*
_output_shapes
:2
sub
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stack
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack_1
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2
strided_slice_2StridedSlicepaddingsstrided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_2n
sub_1SubPartitionedCall:output:1strided_slice_2:output:0*
T0*
_output_shapes
:2
sub_1Q
CastCastsub:z:0*

DstT0*

SrcT0*
_output_shapes
:2
CastI
mulMulspacingCast:y:0*
T0*
_output_shapes
:2
mulW
sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A2	
sub_2/yU
sub_2Submul:z:0sub_2/y:output:0*
T0*
_output_shapes
:2
sub_2P
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yS
addAddV2	sub_1:z:0add/y:output:0*
T0*
_output_shapes
:2
addU
Cast_1Castadd:z:0*

DstT0*

SrcT0*
_output_shapes
:2
Cast_1O
mul_1Mulspacing
Cast_1:y:0*
T0*
_output_shapes
:2
mul_1W
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A2	
add_1/yY
add_1AddV2	mul_1:z:0add_1/y:output:0*
T0*
_output_shapes
:2
add_1P
IdentityIdentity	add_1:z:0*
T0*
_output_shapes
:2

IdentityT

Identity_1Identity	sub_2:z:0*
T0*
_output_shapes
:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*]
_input_shapesL
J:8????????????????????????????????????:::u q
N
_output_shapes<
::8????????????????????????????????????

_user_specified_nameimage:C?

_output_shapes
:
!
_user_specified_name	spacing:HD

_output_shapes

:
"
_user_specified_name
paddings
η	
D
__inference_pad_image_61266	
image
paddings
identity
concat/values_0Const*
_output_shapes

:*
dtype0*)
value B"                2
concat/values_0\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis
concatConcatV2concat/values_0:output:0paddingsconcat/axis:output:0*
N*
T0*
_output_shapes

:2
concatr
PadV2/constant_valuesConst*
_output_shapes
: *
dtype0*
valueB	 jΘ2
PadV2/constant_values±
PadV2PadV2imageconcat:output:0PadV2/constant_values:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2
PadV2
IdentityIdentityPadV2:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:A?????????????????????????????????????????????::~ z
W
_output_shapesE
C:A?????????????????????????????????????????????

_user_specified_nameimage:HD

_output_shapes

:
"
_user_specified_name
paddings
ΌZ
G
__inference_resize_image_61021	
image
new_size
identityU
resize3d/ShapeShapeimage*
T0*
_output_shapes
:2
resize3d/Shape
resize3d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
resize3d/strided_slice/stack
resize3d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
resize3d/strided_slice/stack_1
resize3d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
resize3d/strided_slice/stack_2
resize3d/strided_sliceStridedSliceresize3d/Shape:output:0%resize3d/strided_slice/stack:output:0'resize3d/strided_slice/stack_1:output:0'resize3d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
resize3d/strided_slice
resize3d/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2 
resize3d/strided_slice_1/stack
 resize3d/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_1/stack_1
 resize3d/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_1/stack_2’
resize3d/strided_slice_1StridedSliceresize3d/Shape:output:0'resize3d/strided_slice_1/stack:output:0)resize3d/strided_slice_1/stack_1:output:0)resize3d/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
resize3d/strided_slice_1
resize3d/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2 
resize3d/strided_slice_2/stack
 resize3d/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_2/stack_1
 resize3d/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_2/stack_2
resize3d/strided_slice_2StridedSliceresize3d/Shape:output:0'resize3d/strided_slice_2/stack:output:0)resize3d/strided_slice_2/stack_1:output:0)resize3d/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
resize3d/strided_slice_2
resize3d/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
resize3d/strided_slice_3/stack
 resize3d/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_3/stack_1
 resize3d/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_3/stack_2¬
resize3d/strided_slice_3StridedSlice!resize3d/strided_slice_2:output:0'resize3d/strided_slice_3/stack:output:0)resize3d/strided_slice_3/stack_1:output:0)resize3d/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
resize3d/strided_slice_3
resize3d/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:2 
resize3d/strided_slice_4/stack
 resize3d/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_4/stack_1
 resize3d/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_4/stack_2¬
resize3d/strided_slice_4StridedSlice!resize3d/strided_slice_2:output:0'resize3d/strided_slice_4/stack:output:0)resize3d/strided_slice_4/stack_1:output:0)resize3d/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
resize3d/strided_slice_4
resize3d/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:2 
resize3d/strided_slice_5/stack
 resize3d/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_5/stack_1
 resize3d/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_5/stack_2¬
resize3d/strided_slice_5StridedSlice!resize3d/strided_slice_2:output:0'resize3d/strided_slice_5/stack:output:0)resize3d/strided_slice_5/stack_1:output:0)resize3d/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
resize3d/strided_slice_5
resize3d/transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                2
resize3d/transpose/perm·
resize3d/transpose	Transposeimage resize3d/transpose/perm:output:0*
T0*N
_output_shapes<
::8????????????????????????????????????2
resize3d/transpose
resize3d/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
resize3d/strided_slice_6/stack
 resize3d/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_6/stack_1
 resize3d/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_6/stack_2
resize3d/strided_slice_6StridedSlicenew_size'resize3d/strided_slice_6/stack:output:0)resize3d/strided_slice_6/stack_1:output:0)resize3d/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
resize3d/strided_slice_6
resize3d/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB:2 
resize3d/strided_slice_7/stack
 resize3d/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_7/stack_1
 resize3d/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_7/stack_2
resize3d/strided_slice_7StridedSlicenew_size'resize3d/strided_slice_7/stack:output:0)resize3d/strided_slice_7/stack_1:output:0)resize3d/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
resize3d/strided_slice_7
resize3d/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB:2 
resize3d/strided_slice_8/stack
 resize3d/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_8/stack_1
 resize3d/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_8/stack_2
resize3d/strided_slice_8StridedSlicenew_size'resize3d/strided_slice_8/stack:output:0)resize3d/strided_slice_8/stack_1:output:0)resize3d/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
resize3d/strided_slice_8
resize3d/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
resize3d/Reshape/shape/0ς
resize3d/Reshape/shapePack!resize3d/Reshape/shape/0:output:0!resize3d/strided_slice_4:output:0!resize3d/strided_slice_5:output:0!resize3d/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
resize3d/Reshape/shape½
resize3d/ReshapeReshaperesize3d/transpose:y:0resize3d/Reshape/shape:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
resize3d/Reshape¨
resize3d/resize/sizePack!resize3d/strided_slice_7:output:0!resize3d/strided_slice_8:output:0*
N*
T0*
_output_shapes
:2
resize3d/resize/sizeϋ
resize3d/resize/ResizeBilinearResizeBilinearresize3d/Reshape:output:0resize3d/resize/size:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????*
half_pixel_centers(2 
resize3d/resize/ResizeBilinear»
resize3d/CastCast/resize3d/resize/ResizeBilinear:resized_images:0*

DstT0*

SrcT0*J
_output_shapes8
6:4????????????????????????????????????2
resize3d/Cast
resize3d/Reshape_1/shapePackresize3d/strided_slice:output:0!resize3d/strided_slice_3:output:0!resize3d/strided_slice_7:output:0!resize3d/strided_slice_8:output:0!resize3d/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
resize3d/Reshape_1/shapeΛ
resize3d/Reshape_1Reshaperesize3d/Cast:y:0!resize3d/Reshape_1/shape:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2
resize3d/Reshape_1
resize3d/transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                2
resize3d/transpose_1/permά
resize3d/transpose_1	Transposeresize3d/Reshape_1:output:0"resize3d/transpose_1/perm:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2
resize3d/transpose_1
resize3d/Reshape_2/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
resize3d/Reshape_2/shape/0ψ
resize3d/Reshape_2/shapePack#resize3d/Reshape_2/shape/0:output:0!resize3d/strided_slice_7:output:0!resize3d/strided_slice_3:output:0!resize3d/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
resize3d/Reshape_2/shapeΕ
resize3d/Reshape_2Reshaperesize3d/transpose_1:y:0!resize3d/Reshape_2/shape:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
resize3d/Reshape_2¬
resize3d/resize_1/sizePack!resize3d/strided_slice_7:output:0!resize3d/strided_slice_6:output:0*
N*
T0*
_output_shapes
:2
resize3d/resize_1/size
 resize3d/resize_1/ResizeBilinearResizeBilinearresize3d/Reshape_2:output:0resize3d/resize_1/size:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????*
half_pixel_centers(2"
 resize3d/resize_1/ResizeBilinearΑ
resize3d/Cast_1Cast1resize3d/resize_1/ResizeBilinear:resized_images:0*

DstT0*

SrcT0*J
_output_shapes8
6:4????????????????????????????????????2
resize3d/Cast_1
resize3d/Reshape_3/shapePackresize3d/strided_slice:output:0!resize3d/strided_slice_8:output:0!resize3d/strided_slice_7:output:0!resize3d/strided_slice_6:output:0!resize3d/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
resize3d/Reshape_3/shapeΝ
resize3d/Reshape_3Reshaperesize3d/Cast_1:y:0!resize3d/Reshape_3/shape:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2
resize3d/Reshape_3
resize3d/transpose_2/permConst*
_output_shapes
:*
dtype0*)
value B"                2
resize3d/transpose_2/permά
resize3d/transpose_2	Transposeresize3d/Reshape_3:output:0"resize3d/transpose_2/perm:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2
resize3d/transpose_2
IdentityIdentityresize3d/transpose_2:y:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:8????????????????????????????????????::u q
N
_output_shapes<
::8????????????????????????????????????

_user_specified_nameimage:D@

_output_shapes
:
"
_user_specified_name
new_size
ΌZ
G
__inference_resize_image_62086	
image
new_size
identityU
resize3d/ShapeShapeimage*
T0*
_output_shapes
:2
resize3d/Shape
resize3d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
resize3d/strided_slice/stack
resize3d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
resize3d/strided_slice/stack_1
resize3d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
resize3d/strided_slice/stack_2
resize3d/strided_sliceStridedSliceresize3d/Shape:output:0%resize3d/strided_slice/stack:output:0'resize3d/strided_slice/stack_1:output:0'resize3d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
resize3d/strided_slice
resize3d/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2 
resize3d/strided_slice_1/stack
 resize3d/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_1/stack_1
 resize3d/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_1/stack_2’
resize3d/strided_slice_1StridedSliceresize3d/Shape:output:0'resize3d/strided_slice_1/stack:output:0)resize3d/strided_slice_1/stack_1:output:0)resize3d/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
resize3d/strided_slice_1
resize3d/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2 
resize3d/strided_slice_2/stack
 resize3d/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_2/stack_1
 resize3d/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_2/stack_2
resize3d/strided_slice_2StridedSliceresize3d/Shape:output:0'resize3d/strided_slice_2/stack:output:0)resize3d/strided_slice_2/stack_1:output:0)resize3d/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
resize3d/strided_slice_2
resize3d/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
resize3d/strided_slice_3/stack
 resize3d/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_3/stack_1
 resize3d/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_3/stack_2¬
resize3d/strided_slice_3StridedSlice!resize3d/strided_slice_2:output:0'resize3d/strided_slice_3/stack:output:0)resize3d/strided_slice_3/stack_1:output:0)resize3d/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
resize3d/strided_slice_3
resize3d/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:2 
resize3d/strided_slice_4/stack
 resize3d/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_4/stack_1
 resize3d/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_4/stack_2¬
resize3d/strided_slice_4StridedSlice!resize3d/strided_slice_2:output:0'resize3d/strided_slice_4/stack:output:0)resize3d/strided_slice_4/stack_1:output:0)resize3d/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
resize3d/strided_slice_4
resize3d/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:2 
resize3d/strided_slice_5/stack
 resize3d/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_5/stack_1
 resize3d/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_5/stack_2¬
resize3d/strided_slice_5StridedSlice!resize3d/strided_slice_2:output:0'resize3d/strided_slice_5/stack:output:0)resize3d/strided_slice_5/stack_1:output:0)resize3d/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
resize3d/strided_slice_5
resize3d/transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                2
resize3d/transpose/perm·
resize3d/transpose	Transposeimage resize3d/transpose/perm:output:0*
T0*N
_output_shapes<
::8????????????????????????????????????2
resize3d/transpose
resize3d/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
resize3d/strided_slice_6/stack
 resize3d/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_6/stack_1
 resize3d/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_6/stack_2
resize3d/strided_slice_6StridedSlicenew_size'resize3d/strided_slice_6/stack:output:0)resize3d/strided_slice_6/stack_1:output:0)resize3d/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
resize3d/strided_slice_6
resize3d/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB:2 
resize3d/strided_slice_7/stack
 resize3d/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_7/stack_1
 resize3d/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_7/stack_2
resize3d/strided_slice_7StridedSlicenew_size'resize3d/strided_slice_7/stack:output:0)resize3d/strided_slice_7/stack_1:output:0)resize3d/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
resize3d/strided_slice_7
resize3d/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB:2 
resize3d/strided_slice_8/stack
 resize3d/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_8/stack_1
 resize3d/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_8/stack_2
resize3d/strided_slice_8StridedSlicenew_size'resize3d/strided_slice_8/stack:output:0)resize3d/strided_slice_8/stack_1:output:0)resize3d/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
resize3d/strided_slice_8
resize3d/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
resize3d/Reshape/shape/0ς
resize3d/Reshape/shapePack!resize3d/Reshape/shape/0:output:0!resize3d/strided_slice_4:output:0!resize3d/strided_slice_5:output:0!resize3d/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
resize3d/Reshape/shape½
resize3d/ReshapeReshaperesize3d/transpose:y:0resize3d/Reshape/shape:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
resize3d/Reshape¨
resize3d/resize/sizePack!resize3d/strided_slice_7:output:0!resize3d/strided_slice_8:output:0*
N*
T0*
_output_shapes
:2
resize3d/resize/sizeϋ
resize3d/resize/ResizeBilinearResizeBilinearresize3d/Reshape:output:0resize3d/resize/size:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????*
half_pixel_centers(2 
resize3d/resize/ResizeBilinear»
resize3d/CastCast/resize3d/resize/ResizeBilinear:resized_images:0*

DstT0*

SrcT0*J
_output_shapes8
6:4????????????????????????????????????2
resize3d/Cast
resize3d/Reshape_1/shapePackresize3d/strided_slice:output:0!resize3d/strided_slice_3:output:0!resize3d/strided_slice_7:output:0!resize3d/strided_slice_8:output:0!resize3d/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
resize3d/Reshape_1/shapeΛ
resize3d/Reshape_1Reshaperesize3d/Cast:y:0!resize3d/Reshape_1/shape:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2
resize3d/Reshape_1
resize3d/transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                2
resize3d/transpose_1/permά
resize3d/transpose_1	Transposeresize3d/Reshape_1:output:0"resize3d/transpose_1/perm:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2
resize3d/transpose_1
resize3d/Reshape_2/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
resize3d/Reshape_2/shape/0ψ
resize3d/Reshape_2/shapePack#resize3d/Reshape_2/shape/0:output:0!resize3d/strided_slice_7:output:0!resize3d/strided_slice_3:output:0!resize3d/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
resize3d/Reshape_2/shapeΕ
resize3d/Reshape_2Reshaperesize3d/transpose_1:y:0!resize3d/Reshape_2/shape:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
resize3d/Reshape_2¬
resize3d/resize_1/sizePack!resize3d/strided_slice_7:output:0!resize3d/strided_slice_6:output:0*
N*
T0*
_output_shapes
:2
resize3d/resize_1/size
 resize3d/resize_1/ResizeBilinearResizeBilinearresize3d/Reshape_2:output:0resize3d/resize_1/size:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????*
half_pixel_centers(2"
 resize3d/resize_1/ResizeBilinearΑ
resize3d/Cast_1Cast1resize3d/resize_1/ResizeBilinear:resized_images:0*

DstT0*

SrcT0*J
_output_shapes8
6:4????????????????????????????????????2
resize3d/Cast_1
resize3d/Reshape_3/shapePackresize3d/strided_slice:output:0!resize3d/strided_slice_8:output:0!resize3d/strided_slice_7:output:0!resize3d/strided_slice_6:output:0!resize3d/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
resize3d/Reshape_3/shapeΝ
resize3d/Reshape_3Reshaperesize3d/Cast_1:y:0!resize3d/Reshape_3/shape:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2
resize3d/Reshape_3
resize3d/transpose_2/permConst*
_output_shapes
:*
dtype0*)
value B"                2
resize3d/transpose_2/permά
resize3d/transpose_2	Transposeresize3d/Reshape_3:output:0"resize3d/transpose_2/perm:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2
resize3d/transpose_2
IdentityIdentityresize3d/transpose_2:y:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:8????????????????????????????????????::u q
N
_output_shapes<
::8????????????????????????????????????

_user_specified_nameimage:D@

_output_shapes
:
"
_user_specified_name
new_size
Ή
D
__inference_pad_image_61441	
image
paddings
identity
concat/values_0Const*
_output_shapes

:*
dtype0*)
value B"                2
concat/values_0\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis
concatConcatV2concat/values_0:output:0paddingsconcat/axis:output:0*
N*
T0*
_output_shapes

:2
concat
PadPadimageconcat:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2
Pad
IdentityIdentityPad:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:8????????????????????????????????????::u q
N
_output_shapes<
::8????????????????????????????????????

_user_specified_nameimage:HD

_output_shapes

:
"
_user_specified_name
paddings
θY
G
__inference_resize_image_61958	
image
new_size
identityU
resize3d/ShapeShapeimage*
T0*
_output_shapes
:2
resize3d/Shape
resize3d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
resize3d/strided_slice/stack
resize3d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
resize3d/strided_slice/stack_1
resize3d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
resize3d/strided_slice/stack_2
resize3d/strided_sliceStridedSliceresize3d/Shape:output:0%resize3d/strided_slice/stack:output:0'resize3d/strided_slice/stack_1:output:0'resize3d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
resize3d/strided_slice
resize3d/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2 
resize3d/strided_slice_1/stack
 resize3d/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_1/stack_1
 resize3d/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_1/stack_2’
resize3d/strided_slice_1StridedSliceresize3d/Shape:output:0'resize3d/strided_slice_1/stack:output:0)resize3d/strided_slice_1/stack_1:output:0)resize3d/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
resize3d/strided_slice_1
resize3d/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2 
resize3d/strided_slice_2/stack
 resize3d/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_2/stack_1
 resize3d/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_2/stack_2
resize3d/strided_slice_2StridedSliceresize3d/Shape:output:0'resize3d/strided_slice_2/stack:output:0)resize3d/strided_slice_2/stack_1:output:0)resize3d/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
resize3d/strided_slice_2
resize3d/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
resize3d/strided_slice_3/stack
 resize3d/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_3/stack_1
 resize3d/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_3/stack_2¬
resize3d/strided_slice_3StridedSlice!resize3d/strided_slice_2:output:0'resize3d/strided_slice_3/stack:output:0)resize3d/strided_slice_3/stack_1:output:0)resize3d/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
resize3d/strided_slice_3
resize3d/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:2 
resize3d/strided_slice_4/stack
 resize3d/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_4/stack_1
 resize3d/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_4/stack_2¬
resize3d/strided_slice_4StridedSlice!resize3d/strided_slice_2:output:0'resize3d/strided_slice_4/stack:output:0)resize3d/strided_slice_4/stack_1:output:0)resize3d/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
resize3d/strided_slice_4
resize3d/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:2 
resize3d/strided_slice_5/stack
 resize3d/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_5/stack_1
 resize3d/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_5/stack_2¬
resize3d/strided_slice_5StridedSlice!resize3d/strided_slice_2:output:0'resize3d/strided_slice_5/stack:output:0)resize3d/strided_slice_5/stack_1:output:0)resize3d/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
resize3d/strided_slice_5
resize3d/transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                2
resize3d/transpose/perm·
resize3d/transpose	Transposeimage resize3d/transpose/perm:output:0*
T0*N
_output_shapes<
::8????????????????????????????????????2
resize3d/transpose
resize3d/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
resize3d/strided_slice_6/stack
 resize3d/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_6/stack_1
 resize3d/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_6/stack_2
resize3d/strided_slice_6StridedSlicenew_size'resize3d/strided_slice_6/stack:output:0)resize3d/strided_slice_6/stack_1:output:0)resize3d/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
resize3d/strided_slice_6
resize3d/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB:2 
resize3d/strided_slice_7/stack
 resize3d/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_7/stack_1
 resize3d/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_7/stack_2
resize3d/strided_slice_7StridedSlicenew_size'resize3d/strided_slice_7/stack:output:0)resize3d/strided_slice_7/stack_1:output:0)resize3d/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
resize3d/strided_slice_7
resize3d/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB:2 
resize3d/strided_slice_8/stack
 resize3d/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_8/stack_1
 resize3d/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_8/stack_2
resize3d/strided_slice_8StridedSlicenew_size'resize3d/strided_slice_8/stack:output:0)resize3d/strided_slice_8/stack_1:output:0)resize3d/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
resize3d/strided_slice_8
resize3d/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
resize3d/Reshape/shape/0ς
resize3d/Reshape/shapePack!resize3d/Reshape/shape/0:output:0!resize3d/strided_slice_4:output:0!resize3d/strided_slice_5:output:0!resize3d/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
resize3d/Reshape/shape½
resize3d/ReshapeReshaperesize3d/transpose:y:0resize3d/Reshape/shape:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
resize3d/Reshape¨
resize3d/resize/sizePack!resize3d/strided_slice_7:output:0!resize3d/strided_slice_8:output:0*
N*
T0*
_output_shapes
:2
resize3d/resize/sizeΥ
resize3d/resize/ResizeArea
ResizeArearesize3d/Reshape:output:0resize3d/resize/size:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
resize3d/resize/ResizeArea·
resize3d/CastCast+resize3d/resize/ResizeArea:resized_images:0*

DstT0*

SrcT0*J
_output_shapes8
6:4????????????????????????????????????2
resize3d/Cast
resize3d/Reshape_1/shapePackresize3d/strided_slice:output:0!resize3d/strided_slice_3:output:0!resize3d/strided_slice_7:output:0!resize3d/strided_slice_8:output:0!resize3d/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
resize3d/Reshape_1/shapeΛ
resize3d/Reshape_1Reshaperesize3d/Cast:y:0!resize3d/Reshape_1/shape:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2
resize3d/Reshape_1
resize3d/transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                2
resize3d/transpose_1/permά
resize3d/transpose_1	Transposeresize3d/Reshape_1:output:0"resize3d/transpose_1/perm:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2
resize3d/transpose_1
resize3d/Reshape_2/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
resize3d/Reshape_2/shape/0ψ
resize3d/Reshape_2/shapePack#resize3d/Reshape_2/shape/0:output:0!resize3d/strided_slice_7:output:0!resize3d/strided_slice_3:output:0!resize3d/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
resize3d/Reshape_2/shapeΕ
resize3d/Reshape_2Reshaperesize3d/transpose_1:y:0!resize3d/Reshape_2/shape:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
resize3d/Reshape_2¬
resize3d/resize_1/sizePack!resize3d/strided_slice_7:output:0!resize3d/strided_slice_6:output:0*
N*
T0*
_output_shapes
:2
resize3d/resize_1/sizeέ
resize3d/resize_1/ResizeArea
ResizeArearesize3d/Reshape_2:output:0resize3d/resize_1/size:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
resize3d/resize_1/ResizeArea½
resize3d/Cast_1Cast-resize3d/resize_1/ResizeArea:resized_images:0*

DstT0*

SrcT0*J
_output_shapes8
6:4????????????????????????????????????2
resize3d/Cast_1
resize3d/Reshape_3/shapePackresize3d/strided_slice:output:0!resize3d/strided_slice_8:output:0!resize3d/strided_slice_7:output:0!resize3d/strided_slice_6:output:0!resize3d/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
resize3d/Reshape_3/shapeΝ
resize3d/Reshape_3Reshaperesize3d/Cast_1:y:0!resize3d/Reshape_3/shape:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2
resize3d/Reshape_3
resize3d/transpose_2/permConst*
_output_shapes
:*
dtype0*)
value B"                2
resize3d/transpose_2/permά
resize3d/transpose_2	Transposeresize3d/Reshape_3:output:0"resize3d/transpose_2/perm:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2
resize3d/transpose_2
IdentityIdentityresize3d/transpose_2:y:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:8????????????????????????????????????::u q
N
_output_shapes<
::8????????????????????????????????????

_user_specified_nameimage:D@

_output_shapes
:
"
_user_specified_name
new_size
°
G
!__inference__traced_restore_62155
file_prefix

identity_1€
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2/shape_and_slices°
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapes
:*
dtypes
22
	RestoreV29
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpd
IdentityIdentityfile_prefix^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

IdentityX

Identity_1IdentityIdentity:output:0*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
: :C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ή
D
__inference_pad_image_61094	
image
paddings
identity
concat/values_0Const*
_output_shapes

:*
dtype0*)
value B"                2
concat/values_0\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis
concatConcatV2concat/values_0:output:0paddingsconcat/axis:output:0*
N*
T0*
_output_shapes

:2
concat
PadPadimageconcat:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2
Pad
IdentityIdentityPad:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:8????????????????????????????????????::u q
N
_output_shapes<
::8????????????????????????????????????

_user_specified_nameimage:HD

_output_shapes

:
"
_user_specified_name
paddings
Ψ^
@
__inference_gaussian_60908	
image	
sigma
identitye
gaussian/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @2
gaussian/mul/yf
gaussian/mulMulsigmagaussian/mul/y:output:0*
T0*
_output_shapes
:2
gaussian/mule
gaussian/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
gaussian/add/ys
gaussian/addAddV2gaussian/mul:z:0gaussian/add/y:output:0*
T0*
_output_shapes
:2
gaussian/add[
gaussian/CeilCeilgaussian/add:z:0*
T0*
_output_shapes
:2
gaussian/Ceili
gaussian/mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gaussian/mul_1/yx
gaussian/mul_1Mulgaussian/Ceil:y:0gaussian/mul_1/y:output:0*
T0*
_output_shapes
:2
gaussian/mul_1i
gaussian/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gaussian/add_1/y{
gaussian/add_1AddV2gaussian/mul_1:z:0gaussian/add_1/y:output:0*
T0*
_output_shapes
:2
gaussian/add_1l
gaussian/CastCastgaussian/add_1:z:0*

DstT0*

SrcT0*
_output_shapes
:2
gaussian/Castc
gaussian/Cast_1Castsigma*

DstT0*

SrcT0*
_output_shapes
:2
gaussian/Cast_1
gaussian/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gaussian/strided_slice/stack
gaussian/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
gaussian/strided_slice/stack_1
gaussian/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
gaussian/strided_slice/stack_2
gaussian/strided_sliceStridedSlicegaussian/Cast_1:y:0%gaussian/strided_slice/stack:output:0'gaussian/strided_slice/stack_1:output:0'gaussian/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
shrink_axis_mask2
gaussian/strided_slice
gaussian/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gaussian/strided_slice_1/stack
 gaussian/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 gaussian/strided_slice_1/stack_1
 gaussian/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gaussian/strided_slice_1/stack_2
gaussian/strided_slice_1StridedSlicegaussian/Cast:y:0'gaussian/strided_slice_1/stack:output:0)gaussian/strided_slice_1/stack_1:output:0)gaussian/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
shrink_axis_mask2
gaussian/strided_slice_1ζ
gaussian/PartitionedCallPartitionedCallgaussian/strided_slice:output:0!gaussian/strided_slice_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *,
f'R%
#__inference_gaussian_kernel1d_608562
gaussian/PartitionedCall
gaussian/Reshape/shapeConst*
_output_shapes
:*
dtype0*)
value B"????            2
gaussian/Reshape/shape±
gaussian/ReshapeReshape!gaussian/PartitionedCall:output:0gaussian/Reshape/shape:output:0*
T0*3
_output_shapes!
:?????????2
gaussian/Reshape
gaussian/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2 
gaussian/strided_slice_2/stack
 gaussian/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 gaussian/strided_slice_2/stack_1
 gaussian/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gaussian/strided_slice_2/stack_2 
gaussian/strided_slice_2StridedSlicegaussian/Cast_1:y:0'gaussian/strided_slice_2/stack:output:0)gaussian/strided_slice_2/stack_1:output:0)gaussian/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
shrink_axis_mask2
gaussian/strided_slice_2
gaussian/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:2 
gaussian/strided_slice_3/stack
 gaussian/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 gaussian/strided_slice_3/stack_1
 gaussian/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gaussian/strided_slice_3/stack_2
gaussian/strided_slice_3StridedSlicegaussian/Cast:y:0'gaussian/strided_slice_3/stack:output:0)gaussian/strided_slice_3/stack_1:output:0)gaussian/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
shrink_axis_mask2
gaussian/strided_slice_3μ
gaussian/PartitionedCall_1PartitionedCall!gaussian/strided_slice_2:output:0!gaussian/strided_slice_3:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *,
f'R%
#__inference_gaussian_kernel1d_608562
gaussian/PartitionedCall_1
gaussian/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"   ????         2
gaussian/Reshape_1/shapeΉ
gaussian/Reshape_1Reshape#gaussian/PartitionedCall_1:output:0!gaussian/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:?????????2
gaussian/Reshape_1
gaussian/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:2 
gaussian/strided_slice_4/stack
 gaussian/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 gaussian/strided_slice_4/stack_1
 gaussian/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gaussian/strided_slice_4/stack_2 
gaussian/strided_slice_4StridedSlicegaussian/Cast_1:y:0'gaussian/strided_slice_4/stack:output:0)gaussian/strided_slice_4/stack_1:output:0)gaussian/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
shrink_axis_mask2
gaussian/strided_slice_4
gaussian/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:2 
gaussian/strided_slice_5/stack
 gaussian/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 gaussian/strided_slice_5/stack_1
 gaussian/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gaussian/strided_slice_5/stack_2
gaussian/strided_slice_5StridedSlicegaussian/Cast:y:0'gaussian/strided_slice_5/stack:output:0)gaussian/strided_slice_5/stack_1:output:0)gaussian/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
shrink_axis_mask2
gaussian/strided_slice_5μ
gaussian/PartitionedCall_2PartitionedCall!gaussian/strided_slice_4:output:0!gaussian/strided_slice_5:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *,
f'R%
#__inference_gaussian_kernel1d_608562
gaussian/PartitionedCall_2
gaussian/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*)
value B"      ????      2
gaussian/Reshape_2/shapeΉ
gaussian/Reshape_2Reshape#gaussian/PartitionedCall_2:output:0!gaussian/Reshape_2/shape:output:0*
T0*3
_output_shapes!
:?????????2
gaussian/Reshape_2l
gaussian/floordiv/yConst*
_output_shapes
: *
dtype0*
value	B :2
gaussian/floordiv/y
gaussian/floordivFloorDivgaussian/Cast:y:0gaussian/floordiv/y:output:0*
T0*
_output_shapes
:2
gaussian/floordiv
gaussian/zeros/Rank/ConstConst*
_output_shapes
:*
dtype0*
valueB:2
gaussian/zeros/Rank/Constl
gaussian/zeros/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gaussian/zeros/Rankz
gaussian/zeros/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
gaussian/zeros/range/startz
gaussian/zeros/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
gaussian/zeros/range/deltaΉ
gaussian/zeros/rangeRange#gaussian/zeros/range/start:output:0gaussian/zeros/Rank:output:0#gaussian/zeros/range/delta:output:0*
_output_shapes
:2
gaussian/zeros/range
gaussian/zeros/Prod/inputConst*
_output_shapes
:*
dtype0*
valueB:2
gaussian/zeros/Prod/input
gaussian/zeros/ProdProd"gaussian/zeros/Prod/input:output:0gaussian/zeros/range:output:0*
T0*
_output_shapes
: 2
gaussian/zeros/Prodq
gaussian/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :θ2
gaussian/zeros/Less/y
gaussian/zeros/LessLessgaussian/zeros/Prod:output:0gaussian/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
gaussian/zeros/Less
gaussian/zeros/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:2 
gaussian/zeros/shape_as_tensorn
gaussian/zeros/ConstConst*
_output_shapes
: *
dtype0*
value	B : 2
gaussian/zeros/Const
gaussian/zerosFill'gaussian/zeros/shape_as_tensor:output:0gaussian/zeros/Const:output:0*
T0*
_output_shapes
:2
gaussian/zerosn
gaussian/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
gaussian/concat/axis΄
gaussian/concatConcatV2gaussian/zeros:output:0gaussian/floordiv:z:0gaussian/concat/axis:output:0*
N*
T0*#
_output_shapes
:?????????2
gaussian/concat£
gaussian/stackPackgaussian/concat:output:0gaussian/concat:output:0*
N*
T0*'
_output_shapes
:?????????*

axis2
gaussian/stackΜ
gaussian/MirrorPad	MirrorPadimagegaussian/stack:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????*
mode	SYMMETRIC2
gaussian/MirrorPadv
gaussian/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
gaussian/split/split_dimά
gaussian/splitSplit!gaussian/split/split_dim:output:0gaussian/MirrorPad:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????*
	num_split2
gaussian/splitω
gaussian/conv0Conv3Dgaussian/split:output:0gaussian/Reshape:output:0*
T0*N
_output_shapes<
::8????????????????????????????????????*
data_formatNCDHW*
paddingVALID*
strides	
2
gaussian/conv0?
gaussian/conv0_1Conv3Dgaussian/conv0:output:0gaussian/Reshape_1:output:0*
T0*N
_output_shapes<
::8????????????????????????????????????*
data_formatNCDHW*
paddingVALID*
strides	
2
gaussian/conv0_1
gaussian/conv0_2Conv3Dgaussian/conv0_1:output:0gaussian/Reshape_2:output:0*
T0*N
_output_shapes<
::8????????????????????????????????????*
data_formatNCDHW*
paddingVALID*
strides	
2
gaussian/conv0_2~
gaussian/concat_1/concat_dimConst*
_output_shapes
: *
dtype0*
value	B :2
gaussian/concat_1/concat_dim΄
gaussian/concat_1/concatIdentitygaussian/conv0_2:output:0*
T0*N
_output_shapes<
::8????????????????????????????????????2
gaussian/concat_1/concat
IdentityIdentity!gaussian/concat_1/concat:output:0*
T0*N
_output_shapes<
::8????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:8????????????????????????????????????::u q
N
_output_shapes<
::8????????????????????????????????????

_user_specified_nameimage:?;

_output_shapes
:

_user_specified_namesigma
Ε
x
cond_2_true_60742
cond_2_strided_slice_cast_2
cond_2_placeholder

cond_2_identity
cond_2_identity_1
cond_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
cond_2/strided_slice/stack
cond_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      2
cond_2/strided_slice/stack_1
cond_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
cond_2/strided_slice/stack_2
cond_2/strided_sliceStridedSlicecond_2_strided_slice_cast_2#cond_2/strided_slice/stack:output:0%cond_2/strided_slice/stack_1:output:0%cond_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
cond_2/strided_slice
cond_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"????    2
cond_2/strided_slice_1/stack
cond_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2 
cond_2/strided_slice_1/stack_1
cond_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2 
cond_2/strided_slice_1/stack_2
cond_2/strided_slice_1StridedSlicecond_2_strided_slice_cast_2%cond_2/strided_slice_1/stack:output:0'cond_2/strided_slice_1/stack_1:output:0'cond_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
cond_2/strided_slice_1p
cond_2/IdentityIdentitycond_2/strided_slice_1:output:0*
T0*
_output_shapes
: 2
cond_2/Identityr
cond_2/Identity_1Identitycond_2/strided_slice:output:0*
T0*
_output_shapes
: 2
cond_2/Identity_1"+
cond_2_identitycond_2/Identity:output:0"/
cond_2_identity_1cond_2/Identity_1:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:?????????:'???????????????????????????:- )
'
_output_shapes
:?????????:C?
=
_output_shapes+
):'???????????????????????????
θY
G
__inference_resize_image_61214	
image
new_size
identityU
resize3d/ShapeShapeimage*
T0*
_output_shapes
:2
resize3d/Shape
resize3d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
resize3d/strided_slice/stack
resize3d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
resize3d/strided_slice/stack_1
resize3d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
resize3d/strided_slice/stack_2
resize3d/strided_sliceStridedSliceresize3d/Shape:output:0%resize3d/strided_slice/stack:output:0'resize3d/strided_slice/stack_1:output:0'resize3d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
resize3d/strided_slice
resize3d/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2 
resize3d/strided_slice_1/stack
 resize3d/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_1/stack_1
 resize3d/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_1/stack_2’
resize3d/strided_slice_1StridedSliceresize3d/Shape:output:0'resize3d/strided_slice_1/stack:output:0)resize3d/strided_slice_1/stack_1:output:0)resize3d/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
resize3d/strided_slice_1
resize3d/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2 
resize3d/strided_slice_2/stack
 resize3d/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_2/stack_1
 resize3d/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_2/stack_2
resize3d/strided_slice_2StridedSliceresize3d/Shape:output:0'resize3d/strided_slice_2/stack:output:0)resize3d/strided_slice_2/stack_1:output:0)resize3d/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
resize3d/strided_slice_2
resize3d/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
resize3d/strided_slice_3/stack
 resize3d/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_3/stack_1
 resize3d/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_3/stack_2¬
resize3d/strided_slice_3StridedSlice!resize3d/strided_slice_2:output:0'resize3d/strided_slice_3/stack:output:0)resize3d/strided_slice_3/stack_1:output:0)resize3d/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
resize3d/strided_slice_3
resize3d/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:2 
resize3d/strided_slice_4/stack
 resize3d/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_4/stack_1
 resize3d/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_4/stack_2¬
resize3d/strided_slice_4StridedSlice!resize3d/strided_slice_2:output:0'resize3d/strided_slice_4/stack:output:0)resize3d/strided_slice_4/stack_1:output:0)resize3d/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
resize3d/strided_slice_4
resize3d/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:2 
resize3d/strided_slice_5/stack
 resize3d/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_5/stack_1
 resize3d/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_5/stack_2¬
resize3d/strided_slice_5StridedSlice!resize3d/strided_slice_2:output:0'resize3d/strided_slice_5/stack:output:0)resize3d/strided_slice_5/stack_1:output:0)resize3d/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
resize3d/strided_slice_5
resize3d/transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                2
resize3d/transpose/perm·
resize3d/transpose	Transposeimage resize3d/transpose/perm:output:0*
T0*N
_output_shapes<
::8????????????????????????????????????2
resize3d/transpose
resize3d/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
resize3d/strided_slice_6/stack
 resize3d/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_6/stack_1
 resize3d/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_6/stack_2
resize3d/strided_slice_6StridedSlicenew_size'resize3d/strided_slice_6/stack:output:0)resize3d/strided_slice_6/stack_1:output:0)resize3d/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
resize3d/strided_slice_6
resize3d/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB:2 
resize3d/strided_slice_7/stack
 resize3d/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_7/stack_1
 resize3d/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_7/stack_2
resize3d/strided_slice_7StridedSlicenew_size'resize3d/strided_slice_7/stack:output:0)resize3d/strided_slice_7/stack_1:output:0)resize3d/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
resize3d/strided_slice_7
resize3d/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB:2 
resize3d/strided_slice_8/stack
 resize3d/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_8/stack_1
 resize3d/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_8/stack_2
resize3d/strided_slice_8StridedSlicenew_size'resize3d/strided_slice_8/stack:output:0)resize3d/strided_slice_8/stack_1:output:0)resize3d/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
resize3d/strided_slice_8
resize3d/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
resize3d/Reshape/shape/0ς
resize3d/Reshape/shapePack!resize3d/Reshape/shape/0:output:0!resize3d/strided_slice_4:output:0!resize3d/strided_slice_5:output:0!resize3d/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
resize3d/Reshape/shape½
resize3d/ReshapeReshaperesize3d/transpose:y:0resize3d/Reshape/shape:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
resize3d/Reshape¨
resize3d/resize/sizePack!resize3d/strided_slice_7:output:0!resize3d/strided_slice_8:output:0*
N*
T0*
_output_shapes
:2
resize3d/resize/sizeΥ
resize3d/resize/ResizeArea
ResizeArearesize3d/Reshape:output:0resize3d/resize/size:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
resize3d/resize/ResizeArea·
resize3d/CastCast+resize3d/resize/ResizeArea:resized_images:0*

DstT0*

SrcT0*J
_output_shapes8
6:4????????????????????????????????????2
resize3d/Cast
resize3d/Reshape_1/shapePackresize3d/strided_slice:output:0!resize3d/strided_slice_3:output:0!resize3d/strided_slice_7:output:0!resize3d/strided_slice_8:output:0!resize3d/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
resize3d/Reshape_1/shapeΛ
resize3d/Reshape_1Reshaperesize3d/Cast:y:0!resize3d/Reshape_1/shape:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2
resize3d/Reshape_1
resize3d/transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                2
resize3d/transpose_1/permά
resize3d/transpose_1	Transposeresize3d/Reshape_1:output:0"resize3d/transpose_1/perm:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2
resize3d/transpose_1
resize3d/Reshape_2/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
resize3d/Reshape_2/shape/0ψ
resize3d/Reshape_2/shapePack#resize3d/Reshape_2/shape/0:output:0!resize3d/strided_slice_7:output:0!resize3d/strided_slice_3:output:0!resize3d/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
resize3d/Reshape_2/shapeΕ
resize3d/Reshape_2Reshaperesize3d/transpose_1:y:0!resize3d/Reshape_2/shape:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
resize3d/Reshape_2¬
resize3d/resize_1/sizePack!resize3d/strided_slice_7:output:0!resize3d/strided_slice_6:output:0*
N*
T0*
_output_shapes
:2
resize3d/resize_1/sizeέ
resize3d/resize_1/ResizeArea
ResizeArearesize3d/Reshape_2:output:0resize3d/resize_1/size:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
resize3d/resize_1/ResizeArea½
resize3d/Cast_1Cast-resize3d/resize_1/ResizeArea:resized_images:0*

DstT0*

SrcT0*J
_output_shapes8
6:4????????????????????????????????????2
resize3d/Cast_1
resize3d/Reshape_3/shapePackresize3d/strided_slice:output:0!resize3d/strided_slice_8:output:0!resize3d/strided_slice_7:output:0!resize3d/strided_slice_6:output:0!resize3d/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
resize3d/Reshape_3/shapeΝ
resize3d/Reshape_3Reshaperesize3d/Cast_1:y:0!resize3d/Reshape_3/shape:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2
resize3d/Reshape_3
resize3d/transpose_2/permConst*
_output_shapes
:*
dtype0*)
value B"                2
resize3d/transpose_2/permά
resize3d/transpose_2	Transposeresize3d/Reshape_3:output:0"resize3d/transpose_2/perm:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2
resize3d/transpose_2
IdentityIdentityresize3d/transpose_2:y:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:8????????????????????????????????????::u q
N
_output_shapes<
::8????????????????????????????????????

_user_specified_nameimage:D@

_output_shapes
:
"
_user_specified_name
new_size
Ή
D
__inference_pad_image_61894	
image
paddings
identity
concat/values_0Const*
_output_shapes

:*
dtype0*)
value B"                2
concat/values_0\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis
concatConcatV2concat/values_0:output:0paddingsconcat/axis:output:0*
N*
T0*
_output_shapes

:2
concat
PadPadimageconcat:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2
Pad
IdentityIdentityPad:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:8????????????????????????????????????::u q
N
_output_shapes<
::8????????????????????????????????????

_user_specified_nameimage:HD

_output_shapes

:
"
_user_specified_name
paddings
έ2
I
__inference_bounding_box_61664	
image

identity

identity_1
Any/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Any/reduction_indices]
AnyAnyimageAny/reduction_indices:output:0*#
_output_shapes
:?????????2
AnyN
WhereWhereAny:output:0*'
_output_shapes
:?????????2
Whered
CastCastWhere:index:0*

DstT0*

SrcT0	*'
_output_shapes
:?????????2
CastF
ShapeShapeCast:y:0*
T0*
_output_shapes
:2
ShapeJ
Shape_1ShapeCast:y:0*
T0*
_output_shapes
:2	
Shape_1t
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2δ
strided_sliceStridedSliceShape_1:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceX
	Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2
	Greater/yj
GreaterGreaterstrided_slice:output:0Greater/y:output:0*
T0*
_output_shapes
: 2	
Greater€
condStatelessIfGreater:z:0Cast:y:0image*
Tcond0
*
Tin
2
*
Tout
2*
_lower_using_switch_merge(*
_output_shapes
: : * 
_read_only_resource_inputs
 *#
else_branchR
cond_false_61541*
output_shapes
: : *"
then_branchR
cond_true_615402
condZ
cond/IdentityIdentitycond:output:0*
T0*
_output_shapes
: 2
cond/Identity^
cond/Identity_1Identitycond:output:1*
T0*
_output_shapes
: 2
cond/Identity_1
Any_1/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2
Any_1/reduction_indicesc
Any_1Anyimage Any_1/reduction_indices:output:0*#
_output_shapes
:?????????2
Any_1T
Where_1WhereAny_1:output:0*'
_output_shapes
:?????????2	
Where_1j
Cast_1CastWhere_1:index:0*

DstT0*

SrcT0	*'
_output_shapes
:?????????2
Cast_1L
Shape_2Shape
Cast_1:y:0*
T0*
_output_shapes
:2	
Shape_2L
Shape_3Shape
Cast_1:y:0*
T0*
_output_shapes
:2	
Shape_3x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ξ
strided_slice_1StridedSliceShape_3:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1\
Greater_1/yConst*
_output_shapes
: *
dtype0*
value	B : 2
Greater_1/yr
	Greater_1Greaterstrided_slice_1:output:0Greater_1/y:output:0*
T0*
_output_shapes
: 2
	Greater_1°
cond_1StatelessIfGreater_1:z:0
Cast_1:y:0image*
Tcond0
*
Tin
2
*
Tout
2*
_lower_using_switch_merge(*
_output_shapes
: : * 
_read_only_resource_inputs
 *%
else_branchR
cond_1_false_61585*
output_shapes
: : *$
then_branchR
cond_1_true_615842
cond_1`
cond_1/IdentityIdentitycond_1:output:0*
T0*
_output_shapes
: 2
cond_1/Identityd
cond_1/Identity_1Identitycond_1:output:1*
T0*
_output_shapes
: 2
cond_1/Identity_1
Any_2/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2
Any_2/reduction_indicesc
Any_2Anyimage Any_2/reduction_indices:output:0*#
_output_shapes
:?????????2
Any_2T
Where_2WhereAny_2:output:0*'
_output_shapes
:?????????2	
Where_2j
Cast_2CastWhere_2:index:0*

DstT0*

SrcT0	*'
_output_shapes
:?????????2
Cast_2L
Shape_4Shape
Cast_2:y:0*
T0*
_output_shapes
:2	
Shape_4L
Shape_5Shape
Cast_2:y:0*
T0*
_output_shapes
:2	
Shape_5x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ξ
strided_slice_2StridedSliceShape_5:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2\
Greater_2/yConst*
_output_shapes
: *
dtype0*
value	B : 2
Greater_2/yr
	Greater_2Greaterstrided_slice_2:output:0Greater_2/y:output:0*
T0*
_output_shapes
: 2
	Greater_2°
cond_2StatelessIfGreater_2:z:0
Cast_2:y:0image*
Tcond0
*
Tin
2
*
Tout
2*
_lower_using_switch_merge(*
_output_shapes
: : * 
_read_only_resource_inputs
 *%
else_branchR
cond_2_false_61629*
output_shapes
: : *$
then_branchR
cond_2_true_616282
cond_2`
cond_2/IdentityIdentitycond_2:output:0*
T0*
_output_shapes
: 2
cond_2/Identityd
cond_2/Identity_1Identitycond_2:output:1*
T0*
_output_shapes
: 2
cond_2/Identity_1
stackPackcond/Identity_1:output:0cond_1/Identity_1:output:0cond_2/Identity_1:output:0*
N*
T0*
_output_shapes
:2
stack
stack_1Packcond/Identity:output:0cond_1/Identity:output:0cond_2/Identity:output:0*
N*
T0*
_output_shapes
:2	
stack_1U
IdentityIdentitystack:output:0*
T0*
_output_shapes
:2

Identity[

Identity_1Identitystack_1:output:0*
T0*
_output_shapes
:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:d `
=
_output_shapes+
):'???????????????????????????

_user_specified_nameimage
χ'
ψ
-__inference_preprocess_for_segmentation_61525	
image
spacing
new_spacing
bbox_end

bbox_start	
sigma
valid_sizes_x
valid_sizes_y
valid_sizes_z
identity

identity_1

identity_2

identity_3[
	Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
	Greater/y[
GreaterGreatersigmaGreater/y:output:0*
T0*
_output_shapes
:2	
Greatera
Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 2
Reshape/shapee
Reshape/shape_1Const*
_output_shapes
: *
dtype0*
valueB 2
Reshape/shape_1e
ReshapeReshapeGreater:z:0Reshape/shape_1:output:0*
T0
*
_output_shapes
: 2	
Reshape
condStatelessIfReshape:output:0imagespacingsigma*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*N
_output_shapes<
::8????????????????????????????????????* 
_read_only_resource_inputs
 *#
else_branchR
cond_false_61305*M
output_shapes<
::8????????????????????????????????????*"
then_branchR
cond_true_613042
cond
cond/IdentityIdentitycond:output:0*
T0*N
_output_shapes<
::8????????????????????????????????????2
cond/Identity
PartitionedCallPartitionedCallbbox_end
bbox_startnew_spacingvalid_sizes_xvalid_sizes_yvalid_sizes_z*
Tin

2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:::* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *=
f8R6
4__inference_valid_bbox_extent_for_segmentation_613452
PartitionedCallΕ
PartitionedCall_1PartitionedCallcond/Identity:output:0spacingPartitionedCall:output:0PartitionedCall:output:1*
Tin
2*
Tout
2*
_collective_manager_ids
 *k
_output_shapesY
W:A?????????????????????????????????????????????::* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *1
f,R*
(__inference_crop_and_pad_with_bbox_614462
PartitionedCall_1
PartitionedCall_2PartitionedCallPartitionedCall_1:output:0PartitionedCall:output:2*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:A?????????????????????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *'
f"R 
__inference_resize_image_615132
PartitionedCall_2X
ShapeShapePartitionedCall_1:output:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2Ξ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_sliceψ
PartitionedCall_3PartitionedCallPartitionedCall_2:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:A?????????????????????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *6
f1R/
-__inference_intensity_postprocessing_ct_612842
PartitionedCall_3
IdentityIdentityPartitionedCall_3:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2

Identitya

Identity_1Identitystrided_slice:output:0*
T0*
_output_shapes
:2

Identity_1i

Identity_2IdentityPartitionedCall_1:output:2*
T0*
_output_shapes

:2

Identity_2i

Identity_3IdentityPartitionedCall_1:output:1*
T0*
_output_shapes

:2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:8????????????????????????????????????::::::?????????:?????????:?????????:u q
N
_output_shapes<
::8????????????????????????????????????

_user_specified_nameimage:C?

_output_shapes
:
!
_user_specified_name	spacing:GC

_output_shapes
:
%
_user_specified_namenew_spacing:D@

_output_shapes
:
"
_user_specified_name
bbox_end:FB

_output_shapes
:
$
_user_specified_name
bbox_start:?;

_output_shapes
:

_user_specified_namesigma:RN
#
_output_shapes
:?????????
'
_user_specified_namevalid_sizes_x:RN
#
_output_shapes
:?????????
'
_user_specified_namevalid_sizes_y:RN
#
_output_shapes
:?????????
'
_user_specified_namevalid_sizes_z
σ0
Y
$__inference_center_pad_to_size_61708	
image
size
identity

identity_1C
ShapeShapeimage*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2Ξ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_sliceT
subSubsizestrided_slice:output:0*
T0*
_output_shapes
:2
subZ

floordiv/yConst*
_output_shapes
: *
dtype0*
value	B :2

floordiv/yc
floordivFloorDivsub:z:0floordiv/y:output:0*
T0*
_output_shapes
:2

floordivQ
sub_1Subsub:z:0floordiv:z:0*
T0*
_output_shapes
:2
sub_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2κ
strided_slice_1StridedSlicefloordiv:z:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2η
strided_slice_2StridedSlice	sub_1:z:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2x
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2κ
strided_slice_3StridedSlicefloordiv:z:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3x
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/stack|
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/stack_1|
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/stack_2η
strided_slice_4StridedSlice	sub_1:z:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_4x
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_5/stack|
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_5/stack_1|
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_5/stack_2κ
strided_slice_5StridedSlicefloordiv:z:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_5x
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_6/stack|
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_6/stack_1|
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_6/stack_2η
strided_slice_6StridedSlice	sub_1:z:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_6|
stack/0Packstrided_slice_1:output:0strided_slice_2:output:0*
N*
T0*
_output_shapes
:2	
stack/0|
stack/1Packstrided_slice_3:output:0strided_slice_4:output:0*
N*
T0*
_output_shapes
:2	
stack/1|
stack/2Packstrided_slice_5:output:0strided_slice_6:output:0*
N*
T0*
_output_shapes
:2	
stack/2~
stackPackstack/0:output:0stack/1:output:0stack/2:output:0*
N*
T0*
_output_shapes

:2
stackή
PartitionedCallPartitionedCallimagestack:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:A?????????????????????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *$
fR
__inference_pad_image_612662
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2

Identity]

Identity_1Identitystack:output:0*
T0*
_output_shapes

:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:A?????????????????????????????????????????????::~ z
W
_output_shapesE
C:A?????????????????????????????????????????????

_user_specified_nameimage:@<

_output_shapes
:

_user_specified_namesize
έ

.__inference_postprocess_for_segmentation_61097
segmentation_output
paddings
	croppings
original_cropped_size
identityφ
PartitionedCallPartitionedCallsegmentation_outputoriginal_cropped_size*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:A?????????????????????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *'
f"R 
__inference_resize_image_610212
PartitionedCallf
ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
value	B :2
ArgMax/dimension·
ArgMaxArgMaxPartitionedCall:output:0ArgMax/dimension:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????*
output_type02
ArgMax
CastCastArgMax:output:0*

DstT0*

SrcT0*J
_output_shapes8
6:4????????????????????????????????????2
Castb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim’

ExpandDims
ExpandDimsCast:y:0ExpandDims/dim:output:0*
T0*N
_output_shapes<
::8????????????????????????????????????2

ExpandDimsβ
PartitionedCall_1PartitionedCallExpandDims:output:0paddings*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8????????????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *%
f R
__inference_crop_image_610842
PartitionedCall_1ς
PartitionedCall_2PartitionedCallPartitionedCall_1:output:0	croppings*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:A?????????????????????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *$
fR
__inference_pad_image_610942
PartitionedCall_2
IdentityIdentityPartitionedCall_2:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T:8????????????????????????????????????:::: 
N
_output_shapes<
::8????????????????????????????????????
-
_user_specified_namesegmentation_output:HD

_output_shapes

:
"
_user_specified_name
paddings:IE

_output_shapes

:
#
_user_specified_name	croppings:QM

_output_shapes
:
/
_user_specified_nameoriginal_cropped_size
C
F
__inference_crop_image_61854	
image
	croppings
identityC
ShapeShapeimage*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2Ξ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_1
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2η
strided_slice_1StridedSlice	croppingsstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2τ
strided_slice_2StridedSlicestrided_slice:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_1
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2η
strided_slice_3StridedSlice	croppingsstrided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3f
subSubstrided_slice_2:output:0strided_slice_3:output:0*
T0*
_output_shapes
: 2
sub
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stack
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_4/stack_1
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_4/stack_2η
strided_slice_4StridedSlice	croppingsstrided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_4x
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_5/stack|
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_5/stack_1|
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_5/stack_2τ
strided_slice_5StridedSlicestrided_slice:output:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_5
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_6/stack
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_6/stack_1
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_6/stack_2η
strided_slice_6StridedSlice	croppingsstrided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_6j
sub_1Substrided_slice_5:output:0strided_slice_6:output:0*
T0*
_output_shapes
: 2
sub_1
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_7/stack
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_7/stack_1
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_7/stack_2η
strided_slice_7StridedSlice	croppingsstrided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_7x
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_8/stack|
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_8/stack_1|
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_8/stack_2τ
strided_slice_8StridedSlicestrided_slice:output:0strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_8
strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_9/stack
strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_9/stack_1
strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_9/stack_2η
strided_slice_9StridedSlice	croppingsstrided_slice_9/stack:output:0 strided_slice_9/stack_1:output:0 strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_9j
sub_2Substrided_slice_8:output:0strided_slice_9:output:0*
T0*
_output_shapes
: 2
sub_2v
strided_slice_10/stack/0Const*
_output_shapes
: *
dtype0*
value	B : 2
strided_slice_10/stack/0v
strided_slice_10/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 2
strided_slice_10/stack/1ϊ
strided_slice_10/stackPack!strided_slice_10/stack/0:output:0!strided_slice_10/stack/1:output:0strided_slice_1:output:0strided_slice_4:output:0strided_slice_7:output:0*
N*
T0*
_output_shapes
:2
strided_slice_10/stackz
strided_slice_10/stack_1/0Const*
_output_shapes
: *
dtype0*
value	B : 2
strided_slice_10/stack_1/0z
strided_slice_10/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : 2
strided_slice_10/stack_1/1Σ
strided_slice_10/stack_1Pack#strided_slice_10/stack_1/0:output:0#strided_slice_10/stack_1/1:output:0sub:z:0	sub_1:z:0	sub_2:z:0*
N*
T0*
_output_shapes
:2
strided_slice_10/stack_1
strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*)
value B"               2
strided_slice_10/stack_2ͺ
strided_slice_10StridedSliceimagestrided_slice_10/stack:output:0!strided_slice_10/stack_1:output:0!strided_slice_10/stack_2:output:0*
Index0*
T0*N
_output_shapes<
::8????????????????????????????????????*

begin_mask*
end_mask2
strided_slice_10
IdentityIdentitystrided_slice_10:output:0*
T0*N
_output_shapes<
::8????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:8????????????????????????????????????::u q
N
_output_shapes<
::8????????????????????????????????????

_user_specified_nameimage:IE

_output_shapes

:
#
_user_specified_name	croppings
ρ
l
cond_true_61540
cond_strided_slice_cast
cond_placeholder

cond_identity
cond_identity_1
cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
cond/strided_slice/stack
cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      2
cond/strided_slice/stack_1
cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
cond/strided_slice/stack_2
cond/strided_sliceStridedSlicecond_strided_slice_cast!cond/strided_slice/stack:output:0#cond/strided_slice/stack_1:output:0#cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
cond/strided_slice
cond/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"????    2
cond/strided_slice_1/stack
cond/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
cond/strided_slice_1/stack_1
cond/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
cond/strided_slice_1/stack_2
cond/strided_slice_1StridedSlicecond_strided_slice_cast#cond/strided_slice_1/stack:output:0%cond/strided_slice_1/stack_1:output:0%cond/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
cond/strided_slice_1j
cond/IdentityIdentitycond/strided_slice_1:output:0*
T0*
_output_shapes
: 2
cond/Identityl
cond/Identity_1Identitycond/strided_slice:output:0*
T0*
_output_shapes
: 2
cond/Identity_1"'
cond_identitycond/Identity:output:0"+
cond_identity_1cond/Identity_1:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:?????????:'???????????????????????????:- )
'
_output_shapes
:?????????:C?
=
_output_shapes+
):'???????????????????????????
C
F
__inference_crop_image_61084	
image
	croppings
identityC
ShapeShapeimage*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2Ξ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_1
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2η
strided_slice_1StridedSlice	croppingsstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2τ
strided_slice_2StridedSlicestrided_slice:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_1
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2η
strided_slice_3StridedSlice	croppingsstrided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3f
subSubstrided_slice_2:output:0strided_slice_3:output:0*
T0*
_output_shapes
: 2
sub
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stack
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_4/stack_1
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_4/stack_2η
strided_slice_4StridedSlice	croppingsstrided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_4x
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_5/stack|
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_5/stack_1|
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_5/stack_2τ
strided_slice_5StridedSlicestrided_slice:output:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_5
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_6/stack
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_6/stack_1
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_6/stack_2η
strided_slice_6StridedSlice	croppingsstrided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_6j
sub_1Substrided_slice_5:output:0strided_slice_6:output:0*
T0*
_output_shapes
: 2
sub_1
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_7/stack
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_7/stack_1
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_7/stack_2η
strided_slice_7StridedSlice	croppingsstrided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_7x
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_8/stack|
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_8/stack_1|
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_8/stack_2τ
strided_slice_8StridedSlicestrided_slice:output:0strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_8
strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_9/stack
strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_9/stack_1
strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_9/stack_2η
strided_slice_9StridedSlice	croppingsstrided_slice_9/stack:output:0 strided_slice_9/stack_1:output:0 strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_9j
sub_2Substrided_slice_8:output:0strided_slice_9:output:0*
T0*
_output_shapes
: 2
sub_2v
strided_slice_10/stack/0Const*
_output_shapes
: *
dtype0*
value	B : 2
strided_slice_10/stack/0v
strided_slice_10/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 2
strided_slice_10/stack/1ϊ
strided_slice_10/stackPack!strided_slice_10/stack/0:output:0!strided_slice_10/stack/1:output:0strided_slice_1:output:0strided_slice_4:output:0strided_slice_7:output:0*
N*
T0*
_output_shapes
:2
strided_slice_10/stackz
strided_slice_10/stack_1/0Const*
_output_shapes
: *
dtype0*
value	B : 2
strided_slice_10/stack_1/0z
strided_slice_10/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : 2
strided_slice_10/stack_1/1Σ
strided_slice_10/stack_1Pack#strided_slice_10/stack_1/0:output:0#strided_slice_10/stack_1/1:output:0sub:z:0	sub_1:z:0	sub_2:z:0*
N*
T0*
_output_shapes
:2
strided_slice_10/stack_1
strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*)
value B"               2
strided_slice_10/stack_2ͺ
strided_slice_10StridedSliceimagestrided_slice_10/stack:output:0!strided_slice_10/stack_1:output:0!strided_slice_10/stack_2:output:0*
Index0*
T0*N
_output_shapes<
::8????????????????????????????????????*

begin_mask*
end_mask2
strided_slice_10
IdentityIdentitystrided_slice_10:output:0*
T0*N
_output_shapes<
::8????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:8????????????????????????????????????::u q
N
_output_shapes<
::8????????????????????????????????????

_user_specified_nameimage:IE

_output_shapes

:
#
_user_specified_name	croppings
Ν
l
cond_false_61305
cond_identity_image
cond_placeholder
cond_placeholder_1
cond_identity
cond/IdentityIdentitycond_identity_image*
T0*N
_output_shapes<
::8????????????????????????????????????2
cond/Identity"'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:8????????????????????????????????????:::T P
N
_output_shapes<
::8????????????????????????????????????: 

_output_shapes
::

_output_shapes
:
Ε
x
cond_1_true_61584
cond_1_strided_slice_cast_1
cond_1_placeholder

cond_1_identity
cond_1_identity_1
cond_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
cond_1/strided_slice/stack
cond_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      2
cond_1/strided_slice/stack_1
cond_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
cond_1/strided_slice/stack_2
cond_1/strided_sliceStridedSlicecond_1_strided_slice_cast_1#cond_1/strided_slice/stack:output:0%cond_1/strided_slice/stack_1:output:0%cond_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
cond_1/strided_slice
cond_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"????    2
cond_1/strided_slice_1/stack
cond_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2 
cond_1/strided_slice_1/stack_1
cond_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2 
cond_1/strided_slice_1/stack_2
cond_1/strided_slice_1StridedSlicecond_1_strided_slice_cast_1%cond_1/strided_slice_1/stack:output:0'cond_1/strided_slice_1/stack_1:output:0'cond_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
cond_1/strided_slice_1p
cond_1/IdentityIdentitycond_1/strided_slice_1:output:0*
T0*
_output_shapes
: 2
cond_1/Identityr
cond_1/Identity_1Identitycond_1/strided_slice:output:0*
T0*
_output_shapes
: 2
cond_1/Identity_1"+
cond_1_identitycond_1/Identity:output:0"/
cond_1_identity_1cond_1/Identity_1:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:?????????:'???????????????????????????:- )
'
_output_shapes
:?????????:C?
=
_output_shapes+
):'???????????????????????????
Ά
V
cond_true_61304

cond_image
cond_spacing

cond_sigma
cond_identityς
cond/PartitionedCallPartitionedCall
cond_imagecond_spacing
cond_sigma*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8????????????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *'
f"R 
__inference_smooth_image_611262
cond/PartitionedCall’
cond/IdentityIdentitycond/PartitionedCall:output:0*
T0*N
_output_shapes<
::8????????????????????????????????????2
cond/Identity"'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:8????????????????????????????????????:::T P
N
_output_shapes<
::8????????????????????????????????????: 

_output_shapes
::

_output_shapes
:
ύ
f
cond_false_61541
cond_placeholder
cond_shape_image

cond_identity
cond_identity_1Z

cond/ConstConst*
_output_shapes
: *
dtype0*
value	B : 2

cond/ConstX

cond/ShapeShapecond_shape_image*
T0
*
_output_shapes
:2

cond/Shape~
cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
cond/strided_slice/stack
cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
cond/strided_slice/stack_1
cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
cond/strided_slice/stack_2
cond/strided_sliceStridedSlicecond/Shape:output:0!cond/strided_slice/stack:output:0#cond/strided_slice/stack_1:output:0#cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
cond/strided_sliceZ

cond/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2

cond/sub/yn
cond/subSubcond/strided_slice:output:0cond/sub/y:output:0*
T0*
_output_shapes
: 2

cond/subY
cond/IdentityIdentitycond/sub:z:0*
T0*
_output_shapes
: 2
cond/Identityd
cond/Identity_1Identitycond/Const:output:0*
T0*
_output_shapes
: 2
cond/Identity_1"'
cond_identitycond/Identity:output:0"+
cond_identity_1cond/Identity_1:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:?????????:'???????????????????????????:- )
'
_output_shapes
:?????????:C?
=
_output_shapes+
):'???????????????????????????
Z
G
__inference_resize_image_61513	
image
new_size
identityU
resize3d/ShapeShapeimage*
T0*
_output_shapes
:2
resize3d/Shape
resize3d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
resize3d/strided_slice/stack
resize3d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
resize3d/strided_slice/stack_1
resize3d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
resize3d/strided_slice/stack_2
resize3d/strided_sliceStridedSliceresize3d/Shape:output:0%resize3d/strided_slice/stack:output:0'resize3d/strided_slice/stack_1:output:0'resize3d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
resize3d/strided_slice
resize3d/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2 
resize3d/strided_slice_1/stack
 resize3d/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_1/stack_1
 resize3d/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_1/stack_2’
resize3d/strided_slice_1StridedSliceresize3d/Shape:output:0'resize3d/strided_slice_1/stack:output:0)resize3d/strided_slice_1/stack_1:output:0)resize3d/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
resize3d/strided_slice_1
resize3d/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2 
resize3d/strided_slice_2/stack
 resize3d/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_2/stack_1
 resize3d/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_2/stack_2
resize3d/strided_slice_2StridedSliceresize3d/Shape:output:0'resize3d/strided_slice_2/stack:output:0)resize3d/strided_slice_2/stack_1:output:0)resize3d/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
resize3d/strided_slice_2
resize3d/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
resize3d/strided_slice_3/stack
 resize3d/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_3/stack_1
 resize3d/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_3/stack_2¬
resize3d/strided_slice_3StridedSlice!resize3d/strided_slice_2:output:0'resize3d/strided_slice_3/stack:output:0)resize3d/strided_slice_3/stack_1:output:0)resize3d/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
resize3d/strided_slice_3
resize3d/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:2 
resize3d/strided_slice_4/stack
 resize3d/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_4/stack_1
 resize3d/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_4/stack_2¬
resize3d/strided_slice_4StridedSlice!resize3d/strided_slice_2:output:0'resize3d/strided_slice_4/stack:output:0)resize3d/strided_slice_4/stack_1:output:0)resize3d/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
resize3d/strided_slice_4
resize3d/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:2 
resize3d/strided_slice_5/stack
 resize3d/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_5/stack_1
 resize3d/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_5/stack_2¬
resize3d/strided_slice_5StridedSlice!resize3d/strided_slice_2:output:0'resize3d/strided_slice_5/stack:output:0)resize3d/strided_slice_5/stack_1:output:0)resize3d/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
resize3d/strided_slice_5
resize3d/transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                2
resize3d/transpose/permΐ
resize3d/transpose	Transposeimage resize3d/transpose/perm:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2
resize3d/transpose
resize3d/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
resize3d/strided_slice_6/stack
 resize3d/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_6/stack_1
 resize3d/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_6/stack_2
resize3d/strided_slice_6StridedSlicenew_size'resize3d/strided_slice_6/stack:output:0)resize3d/strided_slice_6/stack_1:output:0)resize3d/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
resize3d/strided_slice_6
resize3d/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB:2 
resize3d/strided_slice_7/stack
 resize3d/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_7/stack_1
 resize3d/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_7/stack_2
resize3d/strided_slice_7StridedSlicenew_size'resize3d/strided_slice_7/stack:output:0)resize3d/strided_slice_7/stack_1:output:0)resize3d/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
resize3d/strided_slice_7
resize3d/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB:2 
resize3d/strided_slice_8/stack
 resize3d/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_8/stack_1
 resize3d/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 resize3d/strided_slice_8/stack_2
resize3d/strided_slice_8StridedSlicenew_size'resize3d/strided_slice_8/stack:output:0)resize3d/strided_slice_8/stack_1:output:0)resize3d/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
resize3d/strided_slice_8
resize3d/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
resize3d/Reshape/shape/0ς
resize3d/Reshape/shapePack!resize3d/Reshape/shape/0:output:0!resize3d/strided_slice_4:output:0!resize3d/strided_slice_5:output:0!resize3d/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
resize3d/Reshape/shape½
resize3d/ReshapeReshaperesize3d/transpose:y:0resize3d/Reshape/shape:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
resize3d/Reshape¨
resize3d/resize/sizePack!resize3d/strided_slice_7:output:0!resize3d/strided_slice_8:output:0*
N*
T0*
_output_shapes
:2
resize3d/resize/sizeΥ
resize3d/resize/ResizeArea
ResizeArearesize3d/Reshape:output:0resize3d/resize/size:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
resize3d/resize/ResizeArea·
resize3d/CastCast+resize3d/resize/ResizeArea:resized_images:0*

DstT0*

SrcT0*J
_output_shapes8
6:4????????????????????????????????????2
resize3d/Cast
resize3d/Reshape_1/shapePackresize3d/strided_slice:output:0!resize3d/strided_slice_3:output:0!resize3d/strided_slice_7:output:0!resize3d/strided_slice_8:output:0!resize3d/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
resize3d/Reshape_1/shapeΛ
resize3d/Reshape_1Reshaperesize3d/Cast:y:0!resize3d/Reshape_1/shape:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2
resize3d/Reshape_1
resize3d/transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                2
resize3d/transpose_1/permά
resize3d/transpose_1	Transposeresize3d/Reshape_1:output:0"resize3d/transpose_1/perm:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2
resize3d/transpose_1
resize3d/Reshape_2/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
resize3d/Reshape_2/shape/0ψ
resize3d/Reshape_2/shapePack#resize3d/Reshape_2/shape/0:output:0!resize3d/strided_slice_7:output:0!resize3d/strided_slice_3:output:0!resize3d/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
resize3d/Reshape_2/shapeΕ
resize3d/Reshape_2Reshaperesize3d/transpose_1:y:0!resize3d/Reshape_2/shape:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
resize3d/Reshape_2¬
resize3d/resize_1/sizePack!resize3d/strided_slice_7:output:0!resize3d/strided_slice_6:output:0*
N*
T0*
_output_shapes
:2
resize3d/resize_1/sizeέ
resize3d/resize_1/ResizeArea
ResizeArearesize3d/Reshape_2:output:0resize3d/resize_1/size:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
resize3d/resize_1/ResizeArea½
resize3d/Cast_1Cast-resize3d/resize_1/ResizeArea:resized_images:0*

DstT0*

SrcT0*J
_output_shapes8
6:4????????????????????????????????????2
resize3d/Cast_1
resize3d/Reshape_3/shapePackresize3d/strided_slice:output:0!resize3d/strided_slice_8:output:0!resize3d/strided_slice_7:output:0!resize3d/strided_slice_6:output:0!resize3d/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
resize3d/Reshape_3/shapeΝ
resize3d/Reshape_3Reshaperesize3d/Cast_1:y:0!resize3d/Reshape_3/shape:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2
resize3d/Reshape_3
resize3d/transpose_2/permConst*
_output_shapes
:*
dtype0*)
value B"                2
resize3d/transpose_2/permά
resize3d/transpose_2	Transposeresize3d/Reshape_3:output:0"resize3d/transpose_2/perm:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2
resize3d/transpose_2
IdentityIdentityresize3d/transpose_2:y:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:A?????????????????????????????????????????????::~ z
W
_output_shapesE
C:A?????????????????????????????????????????????

_user_specified_nameimage:D@

_output_shapes
:
"
_user_specified_name
new_size
Ε
x
cond_1_true_60698
cond_1_strided_slice_cast_1
cond_1_placeholder

cond_1_identity
cond_1_identity_1
cond_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
cond_1/strided_slice/stack
cond_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      2
cond_1/strided_slice/stack_1
cond_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
cond_1/strided_slice/stack_2
cond_1/strided_sliceStridedSlicecond_1_strided_slice_cast_1#cond_1/strided_slice/stack:output:0%cond_1/strided_slice/stack_1:output:0%cond_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
cond_1/strided_slice
cond_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"????    2
cond_1/strided_slice_1/stack
cond_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2 
cond_1/strided_slice_1/stack_1
cond_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2 
cond_1/strided_slice_1/stack_2
cond_1/strided_slice_1StridedSlicecond_1_strided_slice_cast_1%cond_1/strided_slice_1/stack:output:0'cond_1/strided_slice_1/stack_1:output:0'cond_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
cond_1/strided_slice_1p
cond_1/IdentityIdentitycond_1/strided_slice_1:output:0*
T0*
_output_shapes
: 2
cond_1/Identityr
cond_1/Identity_1Identitycond_1/strided_slice:output:0*
T0*
_output_shapes
: 2
cond_1/Identity_1"+
cond_1_identitycond_1/Identity:output:0"/
cond_1_identity_1cond_1/Identity_1:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:?????????:'???????????????????????????:- )
'
_output_shapes
:?????????:C?
=
_output_shapes+
):'???????????????????????????
Κ
p
cond_2_false_61629
cond_2_placeholder
cond_2_shape_image

cond_2_identity
cond_2_identity_1^
cond_2/ConstConst*
_output_shapes
: *
dtype0*
value	B : 2
cond_2/Const^
cond_2/ShapeShapecond_2_shape_image*
T0
*
_output_shapes
:2
cond_2/Shape
cond_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
cond_2/strided_slice/stack
cond_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
cond_2/strided_slice/stack_1
cond_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
cond_2/strided_slice/stack_2
cond_2/strided_sliceStridedSlicecond_2/Shape:output:0#cond_2/strided_slice/stack:output:0%cond_2/strided_slice/stack_1:output:0%cond_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
cond_2/strided_slice^
cond_2/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2
cond_2/sub/yv

cond_2/subSubcond_2/strided_slice:output:0cond_2/sub/y:output:0*
T0*
_output_shapes
: 2

cond_2/sub_
cond_2/IdentityIdentitycond_2/sub:z:0*
T0*
_output_shapes
: 2
cond_2/Identityj
cond_2/Identity_1Identitycond_2/Const:output:0*
T0*
_output_shapes
: 2
cond_2/Identity_1"+
cond_2_identitycond_2/Identity:output:0"/
cond_2_identity_1cond_2/Identity_1:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:?????????:'???????????????????????????:- )
'
_output_shapes
:?????????:C?
=
_output_shapes+
):'???????????????????????????
Κ
p
cond_2_false_60743
cond_2_placeholder
cond_2_shape_image

cond_2_identity
cond_2_identity_1^
cond_2/ConstConst*
_output_shapes
: *
dtype0*
value	B : 2
cond_2/Const^
cond_2/ShapeShapecond_2_shape_image*
T0
*
_output_shapes
:2
cond_2/Shape
cond_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
cond_2/strided_slice/stack
cond_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
cond_2/strided_slice/stack_1
cond_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
cond_2/strided_slice/stack_2
cond_2/strided_sliceStridedSlicecond_2/Shape:output:0#cond_2/strided_slice/stack:output:0%cond_2/strided_slice/stack_1:output:0%cond_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
cond_2/strided_slice^
cond_2/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2
cond_2/sub/yv

cond_2/subSubcond_2/strided_slice:output:0cond_2/sub/y:output:0*
T0*
_output_shapes
: 2

cond_2/sub_
cond_2/IdentityIdentitycond_2/sub:z:0*
T0*
_output_shapes
: 2
cond_2/Identityj
cond_2/Identity_1Identitycond_2/Const:output:0*
T0*
_output_shapes
: 2
cond_2/Identity_1"+
cond_2_identitycond_2/Identity:output:0"/
cond_2_identity_1cond_2/Identity_1:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:?????????:'???????????????????????????:- )
'
_output_shapes
:?????????:C?
=
_output_shapes+
):'???????????????????????????
Ά
V
cond_true_61116

cond_image
cond_spacing

cond_sigma
cond_identityς
cond/PartitionedCallPartitionedCall
cond_imagecond_spacing
cond_sigma*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8????????????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *'
f"R 
__inference_smooth_image_611262
cond/PartitionedCall’
cond/IdentityIdentitycond/PartitionedCall:output:0*
T0*N
_output_shapes<
::8????????????????????????????????????2
cond/Identity"'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:8????????????????????????????????????:::T P
N
_output_shapes<
::8????????????????????????????????????: 

_output_shapes
::

_output_shapes
:
?
Q
__inference_smooth_image_61126	
image
spacing	
sigma
identityP
truedivRealDivsigmaspacing*
T0*
_output_shapes
:2	
truedivΡ
PartitionedCallPartitionedCallimagetruediv:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8????????????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *#
fR
__inference_gaussian_609082
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*N
_output_shapes<
::8????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:8????????????????????????????????????:::u q
N
_output_shapes<
::8????????????????????????????????????

_user_specified_nameimage:C?

_output_shapes
:
!
_user_specified_name	spacing:?;

_output_shapes
:

_user_specified_namesigma
Κ
p
cond_1_false_60699
cond_1_placeholder
cond_1_shape_image

cond_1_identity
cond_1_identity_1^
cond_1/ConstConst*
_output_shapes
: *
dtype0*
value	B : 2
cond_1/Const^
cond_1/ShapeShapecond_1_shape_image*
T0
*
_output_shapes
:2
cond_1/Shape
cond_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
cond_1/strided_slice/stack
cond_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
cond_1/strided_slice/stack_1
cond_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
cond_1/strided_slice/stack_2
cond_1/strided_sliceStridedSlicecond_1/Shape:output:0#cond_1/strided_slice/stack:output:0%cond_1/strided_slice/stack_1:output:0%cond_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
cond_1/strided_slice^
cond_1/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2
cond_1/sub/yv

cond_1/subSubcond_1/strided_slice:output:0cond_1/sub/y:output:0*
T0*
_output_shapes
: 2

cond_1/sub_
cond_1/IdentityIdentitycond_1/sub:z:0*
T0*
_output_shapes
: 2
cond_1/Identityj
cond_1/Identity_1Identitycond_1/Const:output:0*
T0*
_output_shapes
: 2
cond_1/Identity_1"+
cond_1_identitycond_1/Identity:output:0"/
cond_1_identity_1cond_1/Identity_1:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:?????????:'???????????????????????????:- )
'
_output_shapes
:?????????:C?
=
_output_shapes+
):'???????????????????????????

Θ
4__inference_valid_bbox_extent_for_segmentation_61345
bbox_end

bbox_start
spacing
valid_sizes_x
valid_sizes_y
valid_sizes_z
identity

identity_1

identity_2N
addAddV2
bbox_startbbox_end*
T0*
_output_shapes
:2
addS
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
mul/yO
mulMuladd:z:0mul/y:output:0*
T0*
_output_shapes
:2
mulL
subSubbbox_end
bbox_start*
T0*
_output_shapes
:2
subB
CeilCeilsub:z:0*
T0*
_output_shapes
:2
CeilU
truedivRealDivCeil:y:0spacing*
T0*
_output_shapes
:2	
truedivU
CastCasttruediv:z:0*

DstT0*

SrcT0*
_output_shapes
:2
CastΘ
PartitionedCallPartitionedCallCast:y:0valid_sizes_xvalid_sizes_yvalid_sizes_z*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *)
f$R"
 __inference_get_valid_size_609522
PartitionedCallf
Cast_1CastPartitionedCall:output:0*

DstT0*

SrcT0*
_output_shapes
:2
Cast_1O
mul_1Mul
Cast_1:y:0spacing*
T0*
_output_shapes
:2
mul_1]

floordiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2

floordiv/ye
floordivFloorDiv	mul_1:z:0floordiv/y:output:0*
T0*
_output_shapes
:2

floordivQ
sub_1Submul:z:0floordiv:z:0*
T0*
_output_shapes
:2
sub_1a
floordiv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
floordiv_1/yk

floordiv_1FloorDiv	mul_1:z:0floordiv_1/y:output:0*
T0*
_output_shapes
:2

floordiv_1U
add_1AddV2mul:z:0floordiv_1:z:0*
T0*
_output_shapes
:2
add_1P
IdentityIdentity	sub_1:z:0*
T0*
_output_shapes
:2

IdentityT

Identity_1Identity	add_1:z:0*
T0*
_output_shapes
:2

Identity_1c

Identity_2IdentityPartitionedCall:output:0*
T0*
_output_shapes
:2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?::::?????????:?????????:?????????:D @

_output_shapes
:
"
_user_specified_name
bbox_end:FB

_output_shapes
:
$
_user_specified_name
bbox_start:C?

_output_shapes
:
!
_user_specified_name	spacing:RN
#
_output_shapes
:?????????
'
_user_specified_namevalid_sizes_x:RN
#
_output_shapes
:?????????
'
_user_specified_namevalid_sizes_y:RN
#
_output_shapes
:?????????
'
_user_specified_namevalid_sizes_z
Ύ
P
#__inference_gaussian_kernel1d_60856	
sigma
filter_shape
identityB
NegNegfilter_shape*
T0*
_output_shapes
:2
NegZ

floordiv/yConst*
_output_shapes
: *
dtype0*
value	B :2

floordiv/ya
floordivFloorDivNeg:y:0floordiv/y:output:0*
T0*
_output_shapes
:2

floordivP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yT
addAddV2floordiv:z:0add/y:output:0*
T0*
_output_shapes
:2
add^
floordiv_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
floordiv_1/yl

floordiv_1FloorDivfilter_shapefloordiv_1/y:output:0*
T0*
_output_shapes
:2

floordiv_1T
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/y\
add_1AddV2floordiv_1:z:0add_1/y:output:0*
T0*
_output_shapes
:2
add_1\
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range/deltaf
rangeRangeadd:z:0	add_1:z:0range/delta:output:0*#
_output_shapes
:?????????2
rangea
CastCastrange:output:0*

DstT0*

SrcT0*#
_output_shapes
:?????????2
CastR
pow/yConst*
_output_shapes
: *
dtype0*
valueB	 j2
pow/yK
powPowsigmapow/y:output:0*
T0*
_output_shapes
:2
powZ
	truediv/xConst*
_output_shapes
: *
dtype0*
valueB	 jπ2
	truediv/x]
truedivRealDivtruediv/x:output:0pow:z:0*
T0*
_output_shapes
:2	
truedivV
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB	 j2	
pow_1/y_
pow_1PowCast:y:0pow_1/y:output:0*
T0*#
_output_shapes
:?????????2
pow_1L
mulMultruediv:z:0	pow_1:z:0*
T0*
_output_shapes
:2
mul=
ExpExpmul:z:0*
T0*
_output_shapes
:2
Exp>
RankRankExp:y:0*
T0*
_output_shapes
: 2
Rank`
range_1/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range_1/start`
range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range_1/delta
range_1Rangerange_1/start:output:0Rank:output:0range_1/delta:output:0*#
_output_shapes
:?????????2	
range_1M
SumSumExp:y:0range_1:output:0*
T0*
_output_shapes
: 2
Sum[
	truediv_1RealDivExp:y:0Sum:output:0*
T0*
_output_shapes
:2
	truediv_1R
IdentityIdentitytruediv_1:z:0*
T0*
_output_shapes
:2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

:::? ;

_output_shapes
:

_user_specified_namesigma:FB

_output_shapes
:
&
_user_specified_namefilter_shape
έ2
I
__inference_bounding_box_60778	
image

identity

identity_1
Any/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Any/reduction_indices]
AnyAnyimageAny/reduction_indices:output:0*#
_output_shapes
:?????????2
AnyN
WhereWhereAny:output:0*'
_output_shapes
:?????????2
Whered
CastCastWhere:index:0*

DstT0*

SrcT0	*'
_output_shapes
:?????????2
CastF
ShapeShapeCast:y:0*
T0*
_output_shapes
:2
ShapeJ
Shape_1ShapeCast:y:0*
T0*
_output_shapes
:2	
Shape_1t
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2δ
strided_sliceStridedSliceShape_1:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceX
	Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2
	Greater/yj
GreaterGreaterstrided_slice:output:0Greater/y:output:0*
T0*
_output_shapes
: 2	
Greater€
condStatelessIfGreater:z:0Cast:y:0image*
Tcond0
*
Tin
2
*
Tout
2*
_lower_using_switch_merge(*
_output_shapes
: : * 
_read_only_resource_inputs
 *#
else_branchR
cond_false_60655*
output_shapes
: : *"
then_branchR
cond_true_606542
condZ
cond/IdentityIdentitycond:output:0*
T0*
_output_shapes
: 2
cond/Identity^
cond/Identity_1Identitycond:output:1*
T0*
_output_shapes
: 2
cond/Identity_1
Any_1/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2
Any_1/reduction_indicesc
Any_1Anyimage Any_1/reduction_indices:output:0*#
_output_shapes
:?????????2
Any_1T
Where_1WhereAny_1:output:0*'
_output_shapes
:?????????2	
Where_1j
Cast_1CastWhere_1:index:0*

DstT0*

SrcT0	*'
_output_shapes
:?????????2
Cast_1L
Shape_2Shape
Cast_1:y:0*
T0*
_output_shapes
:2	
Shape_2L
Shape_3Shape
Cast_1:y:0*
T0*
_output_shapes
:2	
Shape_3x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ξ
strided_slice_1StridedSliceShape_3:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1\
Greater_1/yConst*
_output_shapes
: *
dtype0*
value	B : 2
Greater_1/yr
	Greater_1Greaterstrided_slice_1:output:0Greater_1/y:output:0*
T0*
_output_shapes
: 2
	Greater_1°
cond_1StatelessIfGreater_1:z:0
Cast_1:y:0image*
Tcond0
*
Tin
2
*
Tout
2*
_lower_using_switch_merge(*
_output_shapes
: : * 
_read_only_resource_inputs
 *%
else_branchR
cond_1_false_60699*
output_shapes
: : *$
then_branchR
cond_1_true_606982
cond_1`
cond_1/IdentityIdentitycond_1:output:0*
T0*
_output_shapes
: 2
cond_1/Identityd
cond_1/Identity_1Identitycond_1:output:1*
T0*
_output_shapes
: 2
cond_1/Identity_1
Any_2/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2
Any_2/reduction_indicesc
Any_2Anyimage Any_2/reduction_indices:output:0*#
_output_shapes
:?????????2
Any_2T
Where_2WhereAny_2:output:0*'
_output_shapes
:?????????2	
Where_2j
Cast_2CastWhere_2:index:0*

DstT0*

SrcT0	*'
_output_shapes
:?????????2
Cast_2L
Shape_4Shape
Cast_2:y:0*
T0*
_output_shapes
:2	
Shape_4L
Shape_5Shape
Cast_2:y:0*
T0*
_output_shapes
:2	
Shape_5x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ξ
strided_slice_2StridedSliceShape_5:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2\
Greater_2/yConst*
_output_shapes
: *
dtype0*
value	B : 2
Greater_2/yr
	Greater_2Greaterstrided_slice_2:output:0Greater_2/y:output:0*
T0*
_output_shapes
: 2
	Greater_2°
cond_2StatelessIfGreater_2:z:0
Cast_2:y:0image*
Tcond0
*
Tin
2
*
Tout
2*
_lower_using_switch_merge(*
_output_shapes
: : * 
_read_only_resource_inputs
 *%
else_branchR
cond_2_false_60743*
output_shapes
: : *$
then_branchR
cond_2_true_607422
cond_2`
cond_2/IdentityIdentitycond_2:output:0*
T0*
_output_shapes
: 2
cond_2/Identityd
cond_2/Identity_1Identitycond_2:output:1*
T0*
_output_shapes
: 2
cond_2/Identity_1
stackPackcond/Identity_1:output:0cond_1/Identity_1:output:0cond_2/Identity_1:output:0*
N*
T0*
_output_shapes
:2
stack
stack_1Packcond/Identity:output:0cond_1/Identity:output:0cond_2/Identity:output:0*
N*
T0*
_output_shapes
:2	
stack_1U
IdentityIdentitystack:output:0*
T0*
_output_shapes
:2

Identity[

Identity_1Identitystack_1:output:0*
T0*
_output_shapes
:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:d `
=
_output_shapes+
):'???????????????????????????

_user_specified_nameimage
ν-

 __inference_get_valid_size_60952
size
valid_output_sizes_x
valid_output_sizes_y
valid_output_size_z
identityt
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2Ψ
strided_sliceStridedSlicesizestrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicey
GreaterGreatervalid_output_sizes_xstrided_slice:output:0*
T0*#
_output_shapes
:?????????2	
Greater
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ς
strided_slice_1StridedSlicevalid_output_sizes_xstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
SelectV2SelectV2Greater:z:0valid_output_sizes_xstrided_slice_1:output:0*
T0*#
_output_shapes
:?????????2

SelectV2X
ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
ConstU
MinMinSelectV2:output:0Const:output:0*
T0*
_output_shapes
: 2
Minx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2β
strided_slice_2StridedSlicesizestrided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2
	Greater_1Greatervalid_output_sizes_ystrided_slice_2:output:0*
T0*#
_output_shapes
:?????????2
	Greater_1
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2ς
strided_slice_3StridedSlicevalid_output_sizes_ystrided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3

SelectV2_1SelectV2Greater_1:z:0valid_output_sizes_ystrided_slice_3:output:0*
T0*#
_output_shapes
:?????????2

SelectV2_1\
Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2	
Const_1]
Min_1MinSelectV2_1:output:0Const_1:output:0*
T0*
_output_shapes
: 2
Min_1x
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/stack|
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/stack_1|
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/stack_2β
strided_slice_4StridedSlicesizestrided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_4~
	Greater_2Greatervalid_output_size_zstrided_slice_4:output:0*
T0*#
_output_shapes
:?????????2
	Greater_2
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_5/stack|
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_5/stack_1|
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_5/stack_2ρ
strided_slice_5StridedSlicevalid_output_size_zstrided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_5

SelectV2_2SelectV2Greater_2:z:0valid_output_size_zstrided_slice_5:output:0*
T0*#
_output_shapes
:?????????2

SelectV2_2\
Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2	
Const_2]
Min_2MinSelectV2_2:output:0Const_2:output:0*
T0*
_output_shapes
: 2
Min_2r
stackPackMin:output:0Min_1:output:0Min_2:output:0*
N*
T0*
_output_shapes
:2
stackU
IdentityIdentitystack:output:0*
T0*
_output_shapes
:2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3::?????????:?????????:?????????:@ <

_output_shapes
:

_user_specified_namesize:YU
#
_output_shapes
:?????????
.
_user_specified_namevalid_output_sizes_x:YU
#
_output_shapes
:?????????
.
_user_specified_namevalid_output_sizes_y:XT
#
_output_shapes
:?????????
-
_user_specified_namevalid_output_size_z
±
H
-__inference_intensity_postprocessing_ct_61284	
image
identityP
add/yConst*
_output_shapes
: *
dtype0*
value	B j 2
add/y
addAddV2imageadd/y:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2
addQ
mul/yConst*
_output_shapes
: *
dtype0*
value
B j 2
mul/y
mulMuladd:z:0mul/y:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2
mulu
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
value
B jx2
clip_by_value/Minimum/yΖ
clip_by_value/MinimumMinimummul:z:0 clip_by_value/Minimum/y:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2
clip_by_value/Minimumf
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB	 jψ2
clip_by_value/yΐ
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2
clip_by_value
IdentityIdentityclip_by_value:z:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A?????????????????????????????????????????????:~ z
W
_output_shapesE
C:A?????????????????????????????????????????????

_user_specified_nameimage

Θ
4__inference_valid_bbox_extent_for_segmentation_62124
bbox_end

bbox_start
spacing
valid_sizes_x
valid_sizes_y
valid_sizes_z
identity

identity_1

identity_2N
addAddV2
bbox_startbbox_end*
T0*
_output_shapes
:2
addS
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
mul/yO
mulMuladd:z:0mul/y:output:0*
T0*
_output_shapes
:2
mulL
subSubbbox_end
bbox_start*
T0*
_output_shapes
:2
subB
CeilCeilsub:z:0*
T0*
_output_shapes
:2
CeilU
truedivRealDivCeil:y:0spacing*
T0*
_output_shapes
:2	
truedivU
CastCasttruediv:z:0*

DstT0*

SrcT0*
_output_shapes
:2
CastΘ
PartitionedCallPartitionedCallCast:y:0valid_sizes_xvalid_sizes_yvalid_sizes_z*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *)
f$R"
 __inference_get_valid_size_609522
PartitionedCallf
Cast_1CastPartitionedCall:output:0*

DstT0*

SrcT0*
_output_shapes
:2
Cast_1O
mul_1Mul
Cast_1:y:0spacing*
T0*
_output_shapes
:2
mul_1]

floordiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2

floordiv/ye
floordivFloorDiv	mul_1:z:0floordiv/y:output:0*
T0*
_output_shapes
:2

floordivQ
sub_1Submul:z:0floordiv:z:0*
T0*
_output_shapes
:2
sub_1a
floordiv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
floordiv_1/yk

floordiv_1FloorDiv	mul_1:z:0floordiv_1/y:output:0*
T0*
_output_shapes
:2

floordiv_1U
add_1AddV2mul:z:0floordiv_1:z:0*
T0*
_output_shapes
:2
add_1P
IdentityIdentity	sub_1:z:0*
T0*
_output_shapes
:2

IdentityT

Identity_1Identity	add_1:z:0*
T0*
_output_shapes
:2

Identity_1c

Identity_2IdentityPartitionedCall:output:0*
T0*
_output_shapes
:2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?::::?????????:?????????:?????????:D @

_output_shapes
:
"
_user_specified_name
bbox_end:FB

_output_shapes
:
$
_user_specified_name
bbox_start:C?

_output_shapes
:
!
_user_specified_name	spacing:RN
#
_output_shapes
:?????????
'
_user_specified_namevalid_sizes_x:RN
#
_output_shapes
:?????????
'
_user_specified_namevalid_sizes_y:RN
#
_output_shapes
:?????????
'
_user_specified_namevalid_sizes_z
ρ
l
cond_true_60654
cond_strided_slice_cast
cond_placeholder

cond_identity
cond_identity_1
cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
cond/strided_slice/stack
cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      2
cond/strided_slice/stack_1
cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
cond/strided_slice/stack_2
cond/strided_sliceStridedSlicecond_strided_slice_cast!cond/strided_slice/stack:output:0#cond/strided_slice/stack_1:output:0#cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
cond/strided_slice
cond/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"????    2
cond/strided_slice_1/stack
cond/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
cond/strided_slice_1/stack_1
cond/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
cond/strided_slice_1/stack_2
cond/strided_slice_1StridedSlicecond_strided_slice_cast#cond/strided_slice_1/stack:output:0%cond/strided_slice_1/stack_1:output:0%cond/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
cond/strided_slice_1j
cond/IdentityIdentitycond/strided_slice_1:output:0*
T0*
_output_shapes
: 2
cond/Identityl
cond/Identity_1Identitycond/strided_slice:output:0*
T0*
_output_shapes
: 2
cond/Identity_1"'
cond_identitycond/Identity:output:0"+
cond_identity_1cond/Identity_1:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:?????????:'???????????????????????????:- )
'
_output_shapes
:?????????:C?
=
_output_shapes+
):'???????????????????????????
Ν
l
cond_false_61117
cond_identity_image
cond_placeholder
cond_placeholder_1
cond_identity
cond/IdentityIdentitycond_identity_image*
T0*N
_output_shapes<
::8????????????????????????????????????2
cond/Identity"'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:8????????????????????????????????????:::T P
N
_output_shapes<
::8????????????????????????????????????: 

_output_shapes
::

_output_shapes
:
±
H
-__inference_intensity_postprocessing_ct_61866	
image
identityP
add/yConst*
_output_shapes
: *
dtype0*
value	B j 2
add/y
addAddV2imageadd/y:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2
addQ
mul/yConst*
_output_shapes
: *
dtype0*
value
B j 2
mul/y
mulMuladd:z:0mul/y:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2
mulu
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
value
B jx2
clip_by_value/Minimum/yΖ
clip_by_value/MinimumMinimummul:z:0 clip_by_value/Minimum/y:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2
clip_by_value/Minimumf
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB	 jψ2
clip_by_value/yΐ
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2
clip_by_value
IdentityIdentityclip_by_value:z:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A?????????????????????????????????????????????:~ z
W
_output_shapesE
C:A?????????????????????????????????????????????

_user_specified_nameimage
Τ
k
__inference__traced_save_62145
file_prefix
savev2_const

identity_1’MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2/shape_and_slicesΊ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0savev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2Ί
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes‘
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
: : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: 
Φ%
Έ
-__inference_preprocess_for_localization_61288	
image
spacing
new_spacing	
sigma
valid_sizes_x
valid_size_y
valid_size_z
identity

identity_1C
ShapeShapeimage*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2Ξ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice[
	Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
	Greater/y[
GreaterGreatersigmaGreater/y:output:0*
T0*
_output_shapes
:2	
Greatera
Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 2
Reshape/shapee
Reshape/shape_1Const*
_output_shapes
: *
dtype0*
valueB 2
Reshape/shape_1e
ReshapeReshapeGreater:z:0Reshape/shape_1:output:0*
T0
*
_output_shapes
: 2	
Reshape
condStatelessIfReshape:output:0imagespacingsigma*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*N
_output_shapes<
::8????????????????????????????????????* 
_read_only_resource_inputs
 *#
else_branchR
cond_false_61117*M
output_shapes<
::8????????????????????????????????????*"
then_branchR
cond_true_611162
cond
cond/IdentityIdentitycond:output:0*
T0*N
_output_shapes<
::8????????????????????????????????????2
cond/Identityΐ
PartitionedCallPartitionedCallstrided_slice:output:0spacingnew_spacing*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *+
f&R$
"__inference_size_for_spacing_611492
PartitionedCall
PartitionedCall_1PartitionedCallcond/Identity:output:0PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:A?????????????????????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *'
f"R 
__inference_resize_image_612142
PartitionedCall_1Ϊ
PartitionedCall_2PartitionedCallPartitionedCall:output:0valid_sizes_xvalid_size_yvalid_size_z*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *)
f$R"
 __inference_get_valid_size_609522
PartitionedCall_2
PartitionedCall_3PartitionedCallPartitionedCall_1:output:0PartitionedCall_2:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *a
_output_shapesO
M:A?????????????????????????????????????????????:* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *-
f(R&
$__inference_center_pad_to_size_612702
PartitionedCall_3ψ
PartitionedCall_4PartitionedCallPartitionedCall_3:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:A?????????????????????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *6
f1R/
-__inference_intensity_postprocessing_ct_612842
PartitionedCall_4
IdentityIdentityPartitionedCall_4:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2

Identityi

Identity_1IdentityPartitionedCall_3:output:1*
T0*
_output_shapes

:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapesy
w:8????????????????????????????????????::::?????????:?????????:?????????:u q
N
_output_shapes<
::8????????????????????????????????????

_user_specified_nameimage:C?

_output_shapes
:
!
_user_specified_name	spacing:GC

_output_shapes
:
%
_user_specified_namenew_spacing:?;

_output_shapes
:

_user_specified_namesigma:RN
#
_output_shapes
:?????????
'
_user_specified_namevalid_sizes_x:QM
#
_output_shapes
:?????????
&
_user_specified_namevalid_size_y:QM
#
_output_shapes
:?????????
&
_user_specified_namevalid_size_z

Z
"__inference_size_for_spacing_62097
size
spacing
new_spacing
identityN
CastCastsize*

DstT0*

SrcT0*
_output_shapes
:2
CastI
mulMulCast:y:0spacing*
T0*
_output_shapes
:2
mulX
truedivRealDivmul:z:0new_spacing*
T0*
_output_shapes
:2	
truedivF
CeilCeiltruediv:z:0*
T0*
_output_shapes
:2
CeilV
Cast_1CastCeil:y:0*

DstT0*

SrcT0*
_output_shapes
:2
Cast_1Q
IdentityIdentity
Cast_1:y:0*
T0*
_output_shapes
:2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
::::@ <

_output_shapes
:

_user_specified_namesize:C?

_output_shapes
:
!
_user_specified_name	spacing:GC

_output_shapes
:
%
_user_specified_namenew_spacing
σ0
Y
$__inference_center_pad_to_size_61270	
image
size
identity

identity_1C
ShapeShapeimage*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2Ξ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_sliceT
subSubsizestrided_slice:output:0*
T0*
_output_shapes
:2
subZ

floordiv/yConst*
_output_shapes
: *
dtype0*
value	B :2

floordiv/yc
floordivFloorDivsub:z:0floordiv/y:output:0*
T0*
_output_shapes
:2

floordivQ
sub_1Subsub:z:0floordiv:z:0*
T0*
_output_shapes
:2
sub_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2κ
strided_slice_1StridedSlicefloordiv:z:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2η
strided_slice_2StridedSlice	sub_1:z:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2x
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2κ
strided_slice_3StridedSlicefloordiv:z:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3x
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/stack|
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/stack_1|
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/stack_2η
strided_slice_4StridedSlice	sub_1:z:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_4x
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_5/stack|
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_5/stack_1|
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_5/stack_2κ
strided_slice_5StridedSlicefloordiv:z:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_5x
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_6/stack|
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_6/stack_1|
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_6/stack_2η
strided_slice_6StridedSlice	sub_1:z:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_6|
stack/0Packstrided_slice_1:output:0strided_slice_2:output:0*
N*
T0*
_output_shapes
:2	
stack/0|
stack/1Packstrided_slice_3:output:0strided_slice_4:output:0*
N*
T0*
_output_shapes
:2	
stack/1|
stack/2Packstrided_slice_5:output:0strided_slice_6:output:0*
N*
T0*
_output_shapes
:2	
stack/2~
stackPackstack/0:output:0stack/1:output:0stack/2:output:0*
N*
T0*
_output_shapes

:2
stackή
PartitionedCallPartitionedCallimagestack:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:A?????????????????????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *$
fR
__inference_pad_image_612662
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2

Identity]

Identity_1Identitystack:output:0*
T0*
_output_shapes

:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:A?????????????????????????????????????????????::~ z
W
_output_shapesE
C:A?????????????????????????????????????????????

_user_specified_nameimage:@<

_output_shapes
:

_user_specified_namesize"ΜJ
saver_filename:0StatefulPartitionedCall:0StatefulPartitionedCall_18"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp:Ί_


signatures
!bbox_from_localization_output
bounding_box
center_pad_to_size
crop_and_pad_with_bbox

crop_image
gaussian
gaussian_kernel1d
	get_valid_size

intensity_postprocessing_ct
	pad_image
 postprocess_for_segmentation
preprocess_for_localization
preprocess_for_segmentation
resize_image
 simulate_localization_output
 simulate_segmentation_output
size_for_spacing
smooth_image
&"valid_bbox_extent_for_segmentation"
_generic_user_object
"
signature_map
Ο2Μ
/__inference_bbox_from_localization_output_60803
―²«
FullArgSpec3
args+(
jself
jimage
	jspacing

jpaddings
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *b’_
?<8????????????????????????????????????



Η2Δ
__inference_bounding_box_61664‘
²
FullArgSpec
args
jself
jimage
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
2
$__inference_center_pad_to_size_61708ά
Σ²Ο
FullArgSpec@
args85
jself
jimage
jsize
j	pad_value
jdata_format
varargs
 
varkw
 %
defaults’
`?
jchannels_last

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
2
(__inference_crop_and_pad_with_bbox_61740Υ
Μ²Θ
FullArgSpecP
argsHE
jself
jimage
	jspacing
j
bbox_start

jbbox_end
jdata_format
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
2
__inference_crop_image_61797
__inference_crop_image_61854Ο
Ζ²Β
FullArgSpec8
args0-
jself
jimage
j	croppings
jdata_format
varargs
 
varkw
  
defaults’
jchannels_last

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
2
__inference_gaussian_60908χ
‘²
FullArgSpec%
args
jself
jimage
jsigma
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *O’L
?<8????????????????????????????????????
	
σ2π
#__inference_gaussian_kernel1d_60856Θ
¨²€
FullArgSpec,
args$!
jself
jsigma
jfilter_shape
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *’
	
	
ί2ά
 __inference_get_valid_size_60952·
ή²Ϊ
FullArgSpecb
argsZW
jself
jsize
jvalid_output_sizes_x
jvalid_output_sizes_y
jvalid_output_size_z
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *R’O

?????????
?????????
?????????
¨2₯
-__inference_intensity_postprocessing_ct_61866σ
κ²ζ
FullArgSpecH
args@=
jself
jimage
jshift
jscale
j	clamp_min
j	clamp_max
varargs
 
varkw
 4
defaults(’%
` 
	Y      @?
	Y      πΏ
	Y      π?

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Ό2Ή
__inference_pad_image_61876
__inference_pad_image_61885
__inference_pad_image_61894ί
Φ²?
FullArgSpecD
args<9
jself
jimage

jpaddings
j	pad_value
jdata_format
varargs
 
varkw
 $
defaults’
` 
jchannels_last

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
2
.__inference_postprocess_for_segmentation_61097?
Ψ²Τ
FullArgSpec\
argsTQ
jself
jsegmentation_output

jpaddings
j	croppings
joriginal_cropped_size
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *s’p
?<8????????????????????????????????????



Υ2?
-__inference_preprocess_for_localization_61288 
μ²θ
FullArgSpecp
argshe
jself
jimage
	jspacing
jnew_spacing
jsigma
jvalid_sizes_x
jvalid_size_y
jvalid_size_z
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *¬’¨
?<8????????????????????????????????????


	
?????????
?????????
?????????
2
-__inference_preprocess_for_segmentation_61525Ω
²
FullArgSpec
args
jself
jimage
	jspacing
jnew_spacing

jbbox_end
j
bbox_start
jsigma
jvalid_sizes_x
jvalid_sizes_y
jvalid_sizes_z
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *Ζ’Β
?<8????????????????????????????????????




	
?????????
?????????
?????????
Ξ2Λ
__inference_resize_image_61958
__inference_resize_image_62022
__inference_resize_image_62086θ
ί²Ϋ
FullArgSpecG
args?<
jself
jimage

jnew_size
jinterpolator
jdata_format
varargs
 
varkw
 *
defaults’
jlinear
jchannels_last

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ί2άΩ
Π²Μ
FullArgSpecT
argsLI
jself
jimage
	jspacing
jnew_spacing
jvalid_sizes
jdata_format
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
2
ό²ψ
FullArgSpecn
argsfc
jself
jimage
	jspacing
jnew_spacing

jbbox_end
j
bbox_start
jvalid_sizes
jdata_format
varargs
 
varkw
  
defaults’
jchannels_last

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
δ2α
"__inference_size_for_spacing_62097Ί
±²­
FullArgSpec5
args-*
jself
jsize
	jspacing
jnew_spacing
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
΅2²
__inference_smooth_image_61126
¬²¨
FullArgSpec0
args(%
jself
jimage
	jspacing
jsigma
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *\’Y
?<8????????????????????????????????????

	
¬2©
4__inference_valid_bbox_extent_for_segmentation_62124π
η²γ
FullArgSpeck
argsc`
jself

jbbox_end
j
bbox_start
	jspacing
jvalid_sizes_x
jvalid_sizes_y
jvalid_sizes_z
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 γ
/__inference_bbox_from_localization_output_60803―’
|’y
FC
image8????????????????????????????????????


spacing

paddings
ͺ "#’ 

0

1
__inference_bounding_box_61664kD’A
:’7
52
image'???????????????????????????

ͺ "#’ 

0

1
$__inference_center_pad_to_size_61708ς’
~’{
OL
imageA?????????????????????????????????????????????

size
`?
jchannels_first
ͺ "d’a
KH
0A?????????????????????????????????????????????

1Ϊ
(__inference_crop_and_pad_with_bbox_61740­°’¬
€’ 
FC
image8????????????????????????????????????

spacing


bbox_start

bbox_end
jchannels_first
ͺ "x’u
KH
0A?????????????????????????????????????????????

1

2ι
__inference_crop_image_61797Θ’
y’v
FC
image8????????????????????????????????????

	croppings
jchannels_first
ͺ "?<8????????????????????????????????????ι
__inference_crop_image_61854Θ’
y’v
FC
image8????????????????????????????????????

	croppings
jchannels_first
ͺ "?<8????????????????????????????????????Ι
__inference_gaussian_60908ͺg’d
]’Z
FC
image8????????????????????????????????????

sigma
ͺ "?<8????????????????????????????????????l
#__inference_gaussian_kernel1d_60856E8’5
.’+

sigma

filter_shape
ͺ "	Ϋ
 __inference_get_valid_size_60952Ά¦’’
’

size
*'
valid_output_sizes_x?????????
*'
valid_output_sizes_y?????????
)&
valid_output_size_z?????????
ͺ "
-__inference_intensity_postprocessing_ct_61866Ρ’
y’v
OL
imageA?????????????????????????????????????????????
` 
	Y      @?
	Y      πΏ
	Y      π?
ͺ "HEA?????????????????????????????????????????????
__inference_pad_image_61876ΰ’
’
OL
imageA?????????????????????????????????????????????

paddings
`?
jchannels_first
ͺ "HEA?????????????????????????????????????????????τ
__inference_pad_image_61885Τ’
|’y
FC
image8????????????????????????????????????

paddings
` 
jchannels_first
ͺ "HEA?????????????????????????????????????????????τ
__inference_pad_image_61894Τ’
|’y
FC
image8????????????????????????????????????

paddings
` 
jchannels_first
ͺ "HEA?????????????????????????????????????????????Α
.__inference_postprocess_for_segmentation_61097Α’½
΅’±
TQ
segmentation_output8????????????????????????????????????

paddings

	croppings
"
original_cropped_size
ͺ "HEA?????????????????????????????????????????????’
-__inference_preprocess_for_localization_61288π’
ϋ’χ
FC
image8????????????????????????????????????

spacing

new_spacing

sigma
# 
valid_sizes_x?????????
"
valid_size_y?????????
"
valid_size_z?????????
ͺ "d’a
KH
0A?????????????????????????????????????????????

1ϊ
-__inference_preprocess_for_segmentation_61525ΘΉ’΅
­’©
FC
image8????????????????????????????????????

spacing

new_spacing

bbox_end


bbox_start

sigma
# 
valid_sizes_x?????????
# 
valid_sizes_y?????????
# 
valid_sizes_z?????????
ͺ "’
KH
0A?????????????????????????????????????????????

1

2

3χ
__inference_resize_image_61958Τ’
|’y
FC
image8????????????????????????????????????

new_size
jarea
jchannels_first
ͺ "HEA?????????????????????????????????????????????
__inference_resize_image_62022ί’
’
OL
imageA?????????????????????????????????????????????

new_size
jarea
jchannels_first
ͺ "HEA?????????????????????????????????????????????ω
__inference_resize_image_62086Φ’
~’{
FC
image8????????????????????????????????????

new_size
jlinear
jchannels_first
ͺ "HEA?????????????????????????????????????????????
"__inference_size_for_spacing_62097_P’M
F’C

size

spacing

new_spacing
ͺ "γ
__inference_smooth_image_61126ΐ}’z
s’p
FC
image8????????????????????????????????????

spacing

sigma
ͺ "?<8????????????????????????????????????Ά
4__inference_valid_bbox_extent_for_segmentation_62124ύΕ’Α
Ή’΅

bbox_end


bbox_start

spacing
# 
valid_sizes_x?????????
# 
valid_sizes_y?????????
# 
valid_sizes_z?????????
ͺ "3’0

0

1

2