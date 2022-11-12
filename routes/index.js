var express = require("express");
var axios = require("axios");
const mobilenet = require("@tensorflow-models/mobilenet");

const tf = require("@tensorflow/tfjs");
const tfcore = require("@tensorflow/tfjs-node");

const image = require("get-image-data");
require('dotenv').config();

var router = express.Router();

/* GET home page. */
router.get("/", function (req, res, next) {
  res.render("index");
});

const imageAPI = {
  api_key: process.env.EXPRESS_APP_IMAGE_API_KEY,
};

function createAPIOptions(query) {
  const options = {
    hostname: "api.pexels.com",
    path: `/v1/search?query=${query}&per_page=1`,
    method: "GET",
  };

  return options;
}

router.get("/classify", async function (req, res, next) {
  let searchTerm = req.query.query;
  const options = createAPIOptions(searchTerm);
  const url = `https://${options.hostname}${options.path}`;
  let config = { Authorization: `${imageAPI.api_key}` };

  await axios
    .get(url, { headers: config })
    .then((response) => {
      return response.data;
    })
    .then((data) => {
      let resultJson = data;

      if (resultJson.total_results > 0) {
        let photoUrl = resultJson.photos[0].src.original;
        let displayMessage = `You searched for ${searchTerm}`;
        // console.log(photoUrl);

        classifyImage(photoUrl)
        .then((imageClassification) => {
          // console.log(imageClassification[0].className);
          res.render("classify", { displayMessage, photoUrl, imageClassification});
          // res.status(200).send({
          //   classification: imageClassification,
          // });
        })
        .catch((err) => {
          console.log(err);
          res.status(500).send("Something went wrong while fetching image from URL.");
        });
      } 

      /* If the query is garbage i.e no photos exist */
      else {
        let displayMessage = "Please enter a valid query";
        res.render("classify" , { displayMessage });
      }
    })
    .catch((err) => {
      let status = err.status;
      let errMessage = err.message;

      res.status(500).render("error", {errMessage, status});
    });
});

function classifyImage(url) {
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
