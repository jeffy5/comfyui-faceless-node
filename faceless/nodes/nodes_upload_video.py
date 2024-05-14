from ufile import filemanager

from .nodes_save_video import NodesSaveVideo
from ..typing import FacelessVideo

class NodesUploadVideo(NodesSaveVideo):

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("FACELESS_VIDEO",),
                "public_key": ("STRING", {
                    "default": "",
                }),
                "private_key": ("STRING", {
                    "default": "",
                }),
                "bucket": ("STRING", {
                    "default": "",
                }),
                "key": ("STRING", {
                    "default": "",
                }),
                "region_host": ("STRING", {
                    "default": ".cn-sh2.ufileos.com",
                }),
            },
        }

    FUNCTION = "upload_video"

    def upload_video(self, video: FacelessVideo, public_key, private_key, bucket, key, region_host):
        super().save_video(video)

        # Uplaod video
        postufile_handler = filemanager.FileManager(public_key, private_key, region_host, region_host)

        # 表单上传文件至空间
        _, resp = postufile_handler.postfile(bucket, key, video["output_path"])
        if resp.status_code != 200:
            raise Exception("Failed to upload video")
        return ()
