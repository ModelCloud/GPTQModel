# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import base64
from io import BytesIO
import numpy as np

try:
    import audioread
    AUDIOREAD_AVAILABLE = True
except ImportError:
    AUDIOREAD_AVAILABLE = False

try:
    import av
    AV_AVAILABLE = True
except ImportError:
    AV_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

AUDIOREAD_INSTALL_HINT = "audioread not installed. Please install via `pip install audioread>=3.1.0`."
AV_INSTALL_HINT = "av not installed. Please install via `pip install av>=16.0.1`."
LIBROSA_INSTALL_HINT = "librosa not installed. Please install via `pip install librosa>=0.11.0`."

def _check_if_video_has_audio(video_path):
    if not AV_AVAILABLE:
        raise ValueError(AV_INSTALL_HINT)
    container = av.open(video_path)
    audio_streams = [stream for stream in container.streams if stream.type == "audio"]
    if not audio_streams:
        return False
    return True


def process_audio_info(conversations: list[dict] | list[list[dict]], 
                       use_audio_in_video: bool, 
                       sample_rate: int=16000
                      ):
    """
    Read and process audio info

    Support dict keys:

    type = audio
    - audio
    - audio_start
    - audio_end

    type = video
    - video
    - video_start
    - video_end
    """
    audios = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    for conversation in conversations:
        for message in conversation:
            if not isinstance(message["content"], list):
                continue
            for ele in message["content"]:
                if ele["type"] == "audio":
                    if "audio" in ele or "audio_url" in ele:
                        path = ele.get("audio", ele.get("audio_url"))
                        audio_start = ele.get("audio_start", 0.0)
                        audio_end = ele.get("audio_end", None)
                        if isinstance(path, np.ndarray):
                            if path.ndim > 1:
                                raise ValueError("Support only mono audio")
                            audios.append(
                                path[int(sample_rate * audio_start) : None if audio_end is None else int(sample_rate * audio_end)]
                            )
                            continue
                        elif path.startswith("data:audio"):
                            _, base64_data = path.split("base64,", 1)
                            data = BytesIO(base64.b64decode(base64_data))
                        elif path.startswith("http://") or path.startswith("https://"):
                            if not AUDIOREAD_AVAILABLE:
                                raise ValueError(AUDIOREAD_INSTALL_HINT)
                            data = audioread.ffdec.FFmpegAudioFile(path)
                        elif path.startswith("file://"):
                            data = path[len("file://") :]
                        else:
                            data = path
                    else:
                        raise ValueError("Unknown audio {}".format(ele))
                elif use_audio_in_video and ele["type"] == "video":
                    if "video" in ele or "video_url" in ele:
                        path = ele.get("video", ele.get("video_url"))
                        audio_start = ele.get("video_start", 0.0)
                        audio_end = ele.get("video_end", None)
                        assert _check_if_video_has_audio(
                            path
                        ), "Video must has audio track when use_audio_in_video=True"
                        if path.startswith("http://") or path.startswith("https://"):
                            if not AUDIOREAD_AVAILABLE:
                                raise ValueError(AUDIOREAD_INSTALL_HINT)
                            data = audioread.ffdec.FFmpegAudioFile(path)
                        elif path.startswith("file://"):
                            data = path[len("file://") :]
                        else:
                            data = path
                    else:
                        raise ValueError("Unknown video {}".format(ele))
                else:
                    continue
                if not LIBROSA_AVAILABLE:
                    raise ValueError(LIBROSA_INSTALL_HINT)
                audios.append(
                    librosa.load(
                        data,
                        sr=sample_rate,
                        offset=audio_start,
                        duration=(audio_end - audio_start) if audio_end is not None else None,
                    )[0]
                )
    if len(audios) == 0:
        audios = None
    return audios
