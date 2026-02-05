#!/bin/bash
#SBATCH --job-name=download_nuscenes_6cam
#SBATCH --partition=lrz-cpu
#SBATCH --qos=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH -o ../logs/%x_%j.out

set -e

NUSCENES_DIR="../data/nuscenes"
BASE_URL="https://d36yt3mvayqw5m.cloudfront.net/public/v1.0"

echo "=========================================="
echo "nuScenes 6-Camera Keyframes Downloader"
echo "Full panoramic view (all 6 cameras)"
echo "=========================================="

# 创建目录
mkdir -p "$NUSCENES_DIR"
cd "$NUSCENES_DIR"

# 检测函数
check_camera_images() {
    local cam=$1
    if [ -d "samples/$cam" ]; then
        local count=$(find "samples/$cam" -name '*.jpg' 2>/dev/null | wc -l)
        echo $count
    else
        echo 0
    fi
}

echo ""
echo "Checking existing data..."

# 检查所有6个摄像头
CAM_FRONT=$(check_camera_images "CAM_FRONT")
CAM_FRONT_LEFT=$(check_camera_images "CAM_FRONT_LEFT")
CAM_FRONT_RIGHT=$(check_camera_images "CAM_FRONT_RIGHT")
CAM_BACK=$(check_camera_images "CAM_BACK")
CAM_BACK_LEFT=$(check_camera_images "CAM_BACK_LEFT")
CAM_BACK_RIGHT=$(check_camera_images "CAM_BACK_RIGHT")

echo "Current status:"
echo "  CAM_FRONT: $CAM_FRONT images"
echo "  CAM_FRONT_LEFT: $CAM_FRONT_LEFT images"
echo "  CAM_FRONT_RIGHT: $CAM_FRONT_RIGHT images"
echo "  CAM_BACK: $CAM_BACK images"
echo "  CAM_BACK_LEFT: $CAM_BACK_LEFT images"
echo "  CAM_BACK_RIGHT: $CAM_BACK_RIGHT images"
echo ""

# 判断是否完成
EXPECTED=34000
ALL_COMPLETE=true
for count in $CAM_FRONT $CAM_FRONT_LEFT $CAM_FRONT_RIGHT $CAM_BACK $CAM_BACK_LEFT $CAM_BACK_RIGHT; do
    if [ "$count" -lt "$EXPECTED" ]; then
        ALL_COMPLETE=false
        break
    fi
done

if [ "$ALL_COMPLETE" = true ]; then
    echo "✓ All 6 cameras already downloaded!"
    echo ""
    echo "Final statistics:"
    ls -lh samples/
    du -sh .
    exit 0
fi

# Step 1: Metadata
if [ ! -d "v1.0-trainval" ]; then
    echo "Step 1: Downloading metadata..."
    if [ ! -f "v1.0-trainval_meta.tgz" ]; then
        wget --continue --progress=bar:force \
             "${BASE_URL}/v1.0-trainval_meta.tgz"
    fi
    
    echo "Extracting metadata..."
    tar -xzf v1.0-trainval_meta.tgz
    rm v1.0-trainval_meta.tgz
    echo "✓ Metadata complete"
else
    echo "✓ Metadata already exists"
fi

# Step 2: Download keyframes (parts 01-10)
echo ""
echo "Step 2: Downloading keyframes with ALL 6 cameras..."
echo "This will take 2-4 hours depending on network speed"
echo "Total size: ~50GB"
echo ""

for i in 1 2 3 4 5 6 7 8 9 10; do
    num=$(printf "%02d" $i)
    tgz_file="v1.0-trainval${num}_keyframes.tgz"
    
    echo "--- Part $num/10 ---"
    
    # 检查是否已解压（通过检查前视图像数量）
    if [ "$CAM_FRONT" -ge $((i * 3400)) ]; then
        echo "✓ Part $num already extracted, skipping..."
        continue
    fi
    
    # 下载（支持断点续传）
    if [ ! -f "$tgz_file" ]; then
        echo "Downloading $tgz_file..."
        wget --continue --progress=bar:force \
             "${BASE_URL}/${tgz_file}" || {
            echo "Download failed, will retry on next run"
            continue
        }
    else
        echo "✓ $tgz_file already downloaded"
    fi
    
    # 解压
    echo "Extracting $tgz_file..."
    tar -xzf "$tgz_file" || {
        echo "Extraction failed, removing corrupted file..."
        rm "$tgz_file"
        continue
    }
    
    # 删除压缩包
    rm "$tgz_file"
    echo "✓ Part $num complete"
    
    # 实时更新计数
    CAM_FRONT=$(check_camera_images "CAM_FRONT")
    echo "Progress: $CAM_FRONT images in CAM_FRONT"
done

# Step 3: Cleanup (only remove LIDAR and sweeps)
echo ""
echo "Step 3: Cleaning up (removing LIDAR/RADAR/sweeps, keeping all 6 cameras)..."
rm -rf samples/LIDAR_TOP 2>/dev/null || true
rm -rf samples/RADAR_* 2>/dev/null || true
rm -rf sweeps 2>/dev/null || true
echo "✓ Cleanup complete"

# Final verification
echo ""
echo "=========================================="
echo "Download Complete!"
echo "=========================================="
echo ""
echo "Data structure:"
ls -lh samples/
echo ""
echo "Image counts per camera:"
for cam in CAM_FRONT CAM_FRONT_LEFT CAM_FRONT_RIGHT CAM_BACK CAM_BACK_LEFT CAM_BACK_RIGHT; do
    count=$(find samples/$cam -name '*.jpg' 2>/dev/null | wc -l)
    echo "  $cam: $count"
done
echo ""
echo "Total size:"
du -sh .
echo ""
echo "Annotation files:"
ls -1 v1.0-trainval/*.json | wc -l | xargs echo "  JSON files:"
echo ""
echo "Ready for 6-camera panoramic training!"