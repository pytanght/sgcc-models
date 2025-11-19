#!/usr/bin/env bash
set -e

# 当前脚本所在目录，假设这里有 detect.py 和 sgcc-models 目录
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_ROOT="$ROOT_DIR/sgcc-models"

#######################
# 1) YOLO 原始输出目录 #
#######################
YOLO_OUT_ROOT="$ROOT_DIR/sgcc-results"
mkdir -p "$YOLO_OUT_ROOT"
mkdir -p "$YOLO_OUT_ROOT/all"

##################################
# 2) 预标注数据集 (LabelImg 用)  #
##################################
PRELABEL_ROOT="$ROOT_DIR/sgcc-prelabel"
IMG_DIR="$PRELABEL_ROOT/images"
LBL_DIR="$PRELABEL_ROOT/labels"
mkdir -p "$IMG_DIR" "$LBL_DIR"

# 定义要跑的模型目录和对应的权重文件名
MODELS=(
  "aj0327:aj0327_yolov5m4.pt"
  "PDZF_yolor:PDZF_yolov5m4.pt"
  "qqzy0.6_yolor:aqzy0.6_yolov5_3.pt"
)

for item in "${MODELS[@]}"; do
  IFS=":" read -r DIR WEIGHT <<< "$item"

  SRC_IMG="$MODEL_ROOT/$DIR/test.jpg"
  WTS="$MODEL_ROOT/$DIR/$WEIGHT"
  NAME="$DIR"   # 用目录名做标签

  if [ ! -f "$SRC_IMG" ]; then
    echo "!!! 找不到测试图片: $SRC_IMG，跳过 $NAME"
    continue
  fi

  # 以“模型名_原始文件名”生成新的图片名，后面 LabelImg 用这个名做基准
  ORI_BASE="$(basename "$SRC_IMG")"          # 例如 test.jpg
  DEST_IMG_BASENAME="${NAME}_${ORI_BASE}"    # 例如 aj0327_test.jpg
  DEST_IMG_PATH="$IMG_DIR/$DEST_IMG_BASENAME"

  # 复制原始图片到预标注数据集目录
  cp "$SRC_IMG" "$DEST_IMG_PATH"

  echo ">>> Running $NAME ..."

  python "$ROOT_DIR/detect.py" \
    --weights "$WTS" \
    --source  "$DEST_IMG_PATH" \
    --img 960 \
    --project "$YOLO_OUT_ROOT" \
    --name "$NAME" \
    --exist-ok \
    --save-txt \
    --save-conf

  # YOLOv5 默认会把结果放到 $YOLO_OUT_ROOT/$NAME 下面
  OUT_SUB="$YOLO_OUT_ROOT/$NAME"

  # 1) 汇总可视化结果到 all/ 目录，文件名前加上模型前缀
  for img in "$OUT_SUB"/*.jpg "$OUT_SUB"/*.png; do
    [ -e "$img" ] || continue
    base="$(basename "$img")"
    cp "$img" "$YOLO_OUT_ROOT/all/${NAME}_$base"
  done

  # 2) 提取 YOLO 预测标签，放到预标注 labels/ 目录
  LABEL_BASENAME="${DEST_IMG_BASENAME%.*}"   # 去掉后缀，aj0327_test
  PRED_LABEL_SRC="$OUT_SUB/labels/${LABEL_BASENAME}.txt"
  PRED_LABEL_DST="$LBL_DIR/${LABEL_BASENAME}.txt"

  if [ -f "$PRED_LABEL_SRC" ]; then
    # 去掉最后一列置信度，只保留前 5 列：class cx cy w h
    awk '{print $1, $2, $3, $4, $5}' "$PRED_LABEL_SRC" > "$PRED_LABEL_DST"
    echo "    预标注标签已生成(去掉 conf): $PRED_LABEL_DST"
  else
    echo "!!! 未找到预测标签文件: $PRED_LABEL_SRC"
  fi

done

echo
echo "所有模型已跑完。"
echo "YOLO 可视化结果:"
echo "  按模型划分: $YOLO_OUT_ROOT/<模型名>/"
echo "  汇总预览:   $YOLO_OUT_ROOT/all/"
echo
echo "预标注数据集目录（后续可直接给 LabelImg / 转换给 Label Studio 用）:"
echo "  图片: $IMG_DIR"
echo "  标签: $LBL_DIR"
echo "  每张图片和标签文件同名，例如: aj0327_test.jpg / aj0327_test.txt"
