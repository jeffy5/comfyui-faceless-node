from .nodes_load_video import NodesLoadVideo
from .nodes_load_frames import NodesLoadFrames
from .nodes_save_video import NodesSaveVideo

from .nodes_face_swap import NodesFaceSwap
from .nodes_face_restore import NodesFaceRestore
from .nodes_video_face_swap import NodesVideoFaceSwap
from .nodes_video_face_restore import NodesVideoFaceRestore

NODE_CLASS_MAPPINGS = {
    "FacelessLoadVideo": NodesLoadVideo,
    "FacelessLoadFrames": NodesLoadFrames,
    "FacelessSaveVideo": NodesSaveVideo,

    "FacelessFaceSwap": NodesFaceSwap,
    "FacelessFaceRestore": NodesFaceRestore,
    "FacelessVideoFaceSwap": NodesVideoFaceSwap,
    "FacelessVideoFaceRestore": NodesVideoFaceRestore,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FacelessLoadVideo": "Load Video",
    "FacelessLoadFrames": "Load Frames",
    "FacelessSaveVideo": "Save Video",

    "FacelessFaceSwap": "Face Swap",
    "FacelessFaceRestore": "Face Restore",
    "FacelessVideoFaceSwap": "Face Swap (Video)",
    "FacelessVideoFaceRestore": "Face Restore (Video)",
}
