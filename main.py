from flask import Flask,request,Response
from flask_cors import CORS,cross_origin
from sarang_utils.graph_utils import graph_class
import base64


app = Flask(__name__)
CORS(app)

class ClientApp:
    def __init__(self):
        self.fileObj = open("Logs.txt","a+")
        self.graph_class = graph_class(self.fileObj)
        self.InputImagePath = "images/Inputimage.jpg"


def DecodeImageIntoBase64(imagestr,imagepath):
    imagedata = base64.b64decode(imagestr)

    with open(imagepath,"wb") as f:
        f.write(imagedata)
        f.close()

def EncodeImageIntoBase64(imagepath):
    with open(imagepath, "rb") as f:
        return base64.b64encode(f.read())


@app.route('/predict',methods=["POST"])
@cross_origin()
def PredictRoute():
    try:
        if request.json['image'] is not None:
            imagestr = request.json['image']
            DecodeImageIntoBase64(imagestr,clientObj.InputImagePath)

            result = clientObj.graph_class.prediction(clientObj.InputImagePath)
            final_result = "NumberPlateDetected : {}".format(result)
            return Response(final_result)
    except Exception as e:
        raise e




if __name__ == "__main__":
    clientObj = ClientApp()
    app.run(debug=True)



