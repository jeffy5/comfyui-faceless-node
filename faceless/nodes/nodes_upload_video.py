import os
import time

from ufile import filemanager

import folder_paths

from ..ffmpeg import merge_video
from ..typing import FacelessVideo

class NodesUploadVideo:

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

    CATEGORY = "faceless"
    RETURN_TYPES = ()
    FUNCTION = "upload_video"
    OUTPUT_NODE = True

    def upload_video(self, video: FacelessVideo, public_key, private_key, bucket, key, region_host):
        video_path = video.get("video_path")
        frames_path = video.get("output_path")
        now = int(time.time())
        output_path = os.path.join(folder_paths.get_output_directory(), "faceless")
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        output_filepath = os.path.join(output_path, f"{now}_" + os.path.basename(video_path))

        resolution = video.get("resolution")
        fps = video.get("fps")

        if not merge_video(frames_path, output_filepath, resolution, fps):
            raise Exception("Failed to merge video")

        # TODO Restore audio

        # Uplaod video
        postufile_handler = filemanager.FileManager(public_key, private_key, region_host, region_host)

        # 表单上传文件至空间
        _, resp = postufile_handler.postfile(bucket, key, output_filepath)
        if resp.status_code != 200:
            raise Exception("Failed to upload video")
        return ()
