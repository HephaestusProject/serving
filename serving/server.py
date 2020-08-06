import sys
import os
from flask import Flask, request
from flask_restplus import Api, Resource, fields

if sys:
    sys.path.insert(0, os.path.dirname(
        os.path.abspath(os.path.dirname(__file__))))
    from handler import Handler

app = Flask(__name__)
api = Api(app,
          version="0.1",
          title="MNIST 모델",
          describtion="MNIST 모델을 추론하는 API입니다")
namespace = api.namespace('mnist', description='MNIST 모델')

handler = Handler()

post_fields = api.model('inference', {
    'base64_image': fields.String,
})


@namespace.route("/model/", endpoint="model")
@namespace.response(200, 'Found')
@namespace.response(404, 'Not found')
@namespace.response(500, 'Internal Error')
class Model(Resource):

    @api.doc("Get Model information")
    def get(self):
        return {'info': 'MNIST Model'}

    @api.doc(params={"base64_image": "image data encoded base64"})
    @api.expect(post_fields, validate=True)
    def post(self):
        inputs = request.get_json()
        pred = handler(inputs)
        return {'pred': pred}


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
