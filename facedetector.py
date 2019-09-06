import glob
import sys
import shutil
import darknet

INPUT_PATH  = "/content/drive/My Drive/darknet/data/face/face_test"
OUTPUT_PATH = "/content/drive/My Drive/darknet/data/face/output"

#同一フォルダへのコピーの禁止
if INPUT_PATH == OUTPUT_PATH:
    print("ERROR: Please choose different directory.")
    sys.exit()

#全画像のパスをリストで受け取り、顔が写っていない画像のパスをリストで返す関数。
def face_detector(paths):

    net = darknet.load_net(b"cfg/face/yolov3-voc.cfg", b"cfg/face/backup/yolov3-voc_900.weights", 0)
    meta = darknet.load_meta(b"cfg/face/datasets.data")

    results = []
    faces = []

    for path in paths:
        r = darknet.detect(net, meta, path.encode())
        results.append(r)

        if len(r) == 0:
            faces.append(path)

    return faces



PICTS_PATH = INPUT_PATH + "/*.jpg"

#INPUT_PATH内の全jpgファイルのフルパスを取得
picts_paths = []
picts_paths = glob.glob(PICTS_PATH)

#顔が写っていない画像のパスのリストを取得
face_paths = face_detector(picts_paths)

#顔が写っていない画像をOUTPUTのフォルダにコピー
for face_path in face_paths:
    shutil.copy2(face_path, OUTPUT_PATH)

print("Done!")
