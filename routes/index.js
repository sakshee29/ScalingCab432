var express = require('express');
const mobilenet = require('@tensorflow-models/mobilenet');

const tf = require("@tensorflow/tfjs");
const tfcore = require("@tensorflow/tfjs-node");

const image = require("get-image-data");


var router = express.Router();

/* GET home page. */
router.get('/image', async function(request, response, next) {

  const requestedWidth = request.query.width ? parseInt(request.query.width) : 700;
  const requestedHeight = request.query.height ? parseInt(request.query.height) : 300;
  const requestedAngle = request.query.angle ? parseInt(request.query.angle) : 30;
  const format = 'jpg';

  console.log(`Handling image: size to ${requestedWidth}x${requestedHeight} and rotation to ${requestedAngle}`);


  sharp('public/images/example.jpg')
    .resize(requestedWidth, requestedHeight, {
      fit: sharp.fit.inside,
    })
    .rotate(requestedAngle)
    .toFormat(format)
    .toBuffer()
    .then(function (outputBuffer) {
      response.type(format);
      response.end(outputBuffer)
    });
  // res.render('index', { title: 'Express' });
});

router.get('/classify', async function(req, res, next) {
  whatIsThis(req.query.url)
    .then((imageClassification) => {
      console.log(imageClassification[0].className);
      res.status(200).send({
        classification: imageClassification,
      });
    })
    .catch((err) => {
      console.log(err);
      res
        .status(500)
        .send("Something went wrong while fetching image from URL.");
    });
});

function whatIsThis(url) {
  return new Promise((resolve, reject) => {
    image(url, async (err, image) => {
      if (err) {
        reject(err);
      } else {
        const channelCount = 3;
        const pixelCount = image.width * image.height;
        const vals = new Int32Array(pixelCount * channelCount);

        let pixels = image.data;

        for (let i = 0; i < pixelCount; i++) {
          for (let k = 0; k < channelCount; k++) {
            vals[i * channelCount + k] = pixels[i * 4 + k];
          }
        }

        const outputShape = [image.height, image.width, channelCount];

        const input = tf.tensor3d(vals, outputShape, "int32");

        const model = await mobilenet.load();

        let temp = await model.classify(input);

        resolve(temp);
      }
    });
  });
}


module.exports = router;
