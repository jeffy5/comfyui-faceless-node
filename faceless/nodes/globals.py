from .nodes_load_video import NodesLoadVideo
from .nodes_load_video_url import NodesLoadVideoUrl
from .nodes_load_image_url import NodesLoadImageUrl
from .nodes_load_frames import NodesLoadFrames
from .nodes_save_video import NodesSaveVideo
from .nodes_merge_videos import NodesMergeVideos
from .nodes_upload_video import NodesUploadVideo

from .nodes_face_swap import NodesFaceSwap
from .nodes_face_restore import NodesFaceRestore
from .nodes_remove_background import NodesRemoveBackground
from .nodes_video_face_swap import NodesVideoFaceSwap
from .nodes_video_face_restore import NodesVideoFaceRestore
from .nodes_video_remove_background import NodesVideoRemoveBackground

NODE_CLASS_MAPPINGS = {
    "FacelessLoadVideo": NodesLoadVideo,
    "FacelessLoadFrames": NodesLoadFrames,
    "FacelessLoadImageUrl": NodesLoadImageUrl,
    "FacelessLoadVideoUrl": NodesLoadVideoUrl,
    "FacelessMergeVideos": NodesMergeVideos,
    "FacelessSaveVideo": NodesSaveVideo,
    "FacelessUploadVideo": NodesUploadVideo,

    "FacelessFaceSwap": NodesFaceSwap,
    "FacelessFaceRestore": NodesFaceRestore,
    "FacelessRemoveBackground": NodesRemoveBackground,

    "FacelessVideoFaceSwap": NodesVideoFaceSwap,
    "FacelessVideoFaceRestore": NodesVideoFaceRestore,
    "FacelessVideoRemoveBackground": NodesVideoRemoveBackground,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FacelessLoadVideo": "Load Video",
    "FacelessLoadFrames": "Load Frames",
    "FacelessLoadImageUrl": "Load Image (Url)",
    "FacelessLoadVideoUrl": "Load Video (Url)",
    "FacelessMergeVideos": "Merge Videos",
    "FacelessSaveVideo": "Save Video",
    "FacelessUploadVideo": "Upload Video",

    "FacelessFaceSwap": "Face Swap",
    "FacelessFaceRestore": "Face Restore",
    "FacelessVideoFaceSwap": "Face Swap (Video)",
    "FacelessVideoFaceRestore": "Face Restore (Video)",
    "FacelessRemoveBackground": "Remove Background",
    "FacelessVideoRemoveBackground": "Remove Background (Video)",
}
