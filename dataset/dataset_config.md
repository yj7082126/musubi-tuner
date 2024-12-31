## sample dataset config file
```toml
# general configurations
[general]
resolution = [960, 544] # optional, [W, H], default is None. This is the default resolution for all datasets
caption_extension = ".txt" # optional, default is None. This is the default caption extension for all datasets
batch_size = 1 # optional, default is 1. This is the default batch size for all datasets
enable_bucket = true # optional, default is false. Enable bucketing for datasets
bucket_no_upscale = false # optional, default is false. Disable upscaling for bucketing. Ignored if enable_bucket is false

# sample image dataset with caption text files
[[datasets]]
image_directory = "/path/to/image_dir"
caption_extension = ".txt" # required for caption text files, if general caption extension is not set
resolution = [960, 544] # required if general resolution is not set
batch_size = 4 # optional, overwrite the default batch size
enable_bucket = false # optional, overwrite the default bucketing setting
bucket_no_upscale = true # optional, overwrite the default bucketing setting
cache_directory = "/path/to/cache_directory" # optional, default is None to use the same directory as the image directory. NOTE: caching is always enabled

# sample image dataset with metadata **jsonl** file
[[datasets]]
image_jsonl_file = "/path/to/metadata.jsonl" # includes pairs of image files and captions
resolution = [960, 544] # required if general resolution is not set
cache_directory = "/path/to/cache_directory" # required for metadata jsonl file
# caption_extension is not required for metadata jsonl file
# batch_size, enable_bucket, bucket_no_upscale are also available for metadata jsonl file

# sample video dataset with caption text files
[[datasets]]
video_directory = "/path/to/video_dir"
caption_extension = ".txt" # required for caption text files, if general caption extension is not set
resolution = [960, 544] # required if general resolution is not set
target_frames = [1, 25, 79] # required for video dataset. list of video lengths to extract frames. each element must be N*4+1 (N=0,1,2,...)
frame_extraction = "head" # optional, "head" or "chunk", "slide", "uniform". Default is "head"
frame_stride = 1 # optional, default is 1, available for "slide" frame extraction
frame_sample = 4 # optional, default is 1 (same as "head"), available for "uniform" frame extraction
# batch_size, enable_bucket, bucket_no_upscale, cache_directory are also available for video dataset

# sample video dataset with metadata jsonl file
[[datasets]]
video_jsonl_file = "/path/to/metadata.jsonl" # includes pairs of video files and captions
target_frames = [1, 79]
cache_directory = "/path/to/cache_directory" # required for metadata jsonl file
# frame_extraction, frame_stride, frame_sample are also available for metadata jsonl file
```

<!-- 
# sample image dataset with lance
[[datasets]]
image_lance_dataset = "/path/to/lance_dataset"
resolution = [960, 544] # required if general resolution is not set
# batch_size, enable_bucket, bucket_no_upscale, cache_directory are also available for lance dataset
-->

The metadata with .json file will be supported in the near future.



