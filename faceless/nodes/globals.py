from .nodes_load_video import NodesLoadVideo
from .nodes_load_video_url import NodesLoadVideoUrl
from .nodes_load_image_url import NodesLoadImageUrl
from .nodes_load_frames import NodesLoadFrames
from .nodes_save_video import NodesSaveVideo
from .nodes_upload_video import NodesUploadVideo

from .nodes_face_swap import NodesFaceSwap
from .nodes_face_restore import NodesFaceRestore
from .nodes_video_face_swap import NodesVideoFaceSwap
from .nodes_video_face_restore import NodesVideoFaceRestore

NODE_CLASS_MAPPINGS = {
    "FacelessLoadVideo": NodesLoadVideo,
    "FacelessLoadFrames": NodesLoadFrames,
    "FacelessSaveVideo": NodesSaveVideo,
    "FacelessLoadImageUrl": NodesLoadImageUrl,
    "FacelessLoadVideoUrl": NodesLoadVideoUrl,
    "FacelessUploadVideo": NodesUploadVideo,

    "FacelessFaceSwap": NodesFaceSwap,
    "FacelessFaceRestore": NodesFaceRestore,
    "FacelessVideoFaceSwap": NodesVideoFaceSwap,
    "FacelessVideoFaceRestore": NodesVideoFaceRestore,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FacelessLoadVideo": "Load Video",
    "FacelessLoadFrames": "Load Frames",
    "FacelessSaveVideo": "Save Video",
    "FacelessLoadImageUrl": "Load Image (Url)",
    "FacelessLoadVideoUrl": "Load Video (Url)",
    "FacelessUploadVideo": "Upload Video",

    "FacelessFaceSwap": "Face Swap",
    "FacelessFaceRestore": "Face Restore",
    "FacelessVideoFaceSwap": "Face Swap (Video)",
    "FacelessVideoFaceRestore": "Face Restore (Video)",
}
