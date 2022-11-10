var express = require('express');
var axios = require("axios");
const mobilenet = require('@tensorflow-models/mobilenet');

const tf = require("@tensorflow/tfjs");
const tfcore = require("@tensorflow/tfjs-node");

const image = require("get-image-data");
const { response } = require('express');


var router = express.Router();

/* GET home page. */
router.get("/", function(req, res, next) {
  res.render("index");
});

const imageAPI = {
  api_key: "563492ad6f917000010000014e2e8fed9d1d45b2b612e94a91a62e94",
};

function createAPIOptions(query) {
  const options = {
    hostname: "api.pexels.com",
    path: `/v1/search?query=${query}&per_page=1`,
    method: "GET",
  };

  return options;
}



router.get('/classify', async function(req, res, next) {

  const options = createAPIOptions(req.query.query);
  const url = `https://${options.hostname}${options.path}`;
  let config = {'Authorization': `${imageAPI.api_key}`};

  await axios
    .get(url, {headers : config})
    .then((response) => {
      console.log(response.status);
      return response.data;
    })
    .then((data) => {
      let resultJson = data;
      /* If the query is garbage i.e no photos exist */
      if(resultJson.total_results > 0){
        res.status(200).send({
          result: resultJson,
        });
      }
      else{
        res.status(200).send("Please enter a valid query");
      }
      
    })
    .catch((err) => {
      let status = 500;
      let errMessage = err.message;

      if (err.response) {
        /*If query is empty */
        status = err.response.data.status;
        errMessage = err.response.data.code; 
        // console.log(err.response.data);
      }
      res.status(status).send(errMessage);
    })


  // whatIsThis(req.query.query)
  //   .then((imageClassification) => {
  //     console.log(imageClassification[0].className);
  //     res.status(200).send({
  //       classification: imageClassification,
  //     });
  //   })
  //   .catch((err) => {
  //     console.log(err);
  //     res
  //       .status(500)
  //       .send("Something went wrong while fetching image from URL.");
  //   });
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

// router.get('/wiki', async function(req, res, next) {
//   const searchQuery = req.query.search
//   const searchUrl = `https://en.wikipedia.org/w/api.php?action=parse&format=json&section=0&page=${searchQuery}`;

//   await axios
//     .get(searchUrl)
//     .then((response) => {
//       const responseJSON = response.data;
//       res.status(200).send({
//         response: responseJSON,
//       });
//     })
//     .catch((err) => res.json(err));
// });

module.exports = router;
