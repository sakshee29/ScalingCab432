const express = require("express");
const axios = require("axios");
const mobilenet = require("@tensorflow-models/mobilenet");
const tf = require("@tensorflow/tfjs");
//const tfcore = require("@tensorflow/tfjs-node");
const image = require("get-image-data");
const redis = require("redis");

var router = express.Router();

require("dotenv").config();
const AWS = require("aws-sdk");

/* ------------------------ Cloud Services Set-up  ------------------------ */
/* Create unique bucket name (S3) */
const bucketName = "a2-sak-test";
const s3 = new AWS.S3({ apiVersion: "2006-03-01" });

s3.createBucket({ Bucket: bucketName })
  .promise()
  .then(() => console.log(`Created bucket: ${bucketName}`))
  .catch((err) => {
    // Ignore 409 errors which indicate that the bucket already exists
    if (err.statusCode !== 409) {
      console.log(`Error creating bucket: ${err}`);
    }
  });

/* Redis setup */
const redisClient = redis.createClient();
redisClient.connect().catch((err) => {
  console.log(err);
});

/* ------------------------ Routes ------------------------ */

/* GET home page. */
router.get("/", function (req, res, next) {
  res.render("index");
});

/* GET classification results page. */
router.get("/classify", async function (req, res, next) {
  let searchTerm = req.query.query;
  const options = createAPIOptions(searchTerm);
  const url = `https://${options.hostname}${options.path}`;
  let config = { Authorization: `${process.env.EXPRESS_APP_IMAGE_API_KEY}` };

  await axios
    .get(url, { headers: config })
    .then((response) => {
      return response.data;
    })
    .then((data) => {
      let resultJson = data;

      if (resultJson.total_results > 0) {
        let photoUrl = resultJson.photos[0].src.original;
        // console.log(photoUrl);
        let displayMessage = `You searched for ${searchTerm}`;
        const persistenceKey = `${photoUrl}`;

        redisClient.get(persistenceKey).then((result) => {
          if (result) {
            /* Serve from Redis */
            const resultJSON = JSON.parse(result);
            let prob = (
              resultJSON[0].probability * 100
            ).toFixed(2);
            // res.json({ source: "Redis Cache", ...resultJSON });
            res.render("classify", { displayMessage, photoUrl, prob, imageClassification: resultJSON, source: "Redis Cache" });
    
          } 
          else {
            /* Check if it's in S3 */
            const params = { Bucket: bucketName, Key: persistenceKey };
            s3.getObject(params)
              .promise()
              .then((result) => {
                /* Serve from S3 and store in redis */
                const resultJSON = JSON.parse(result.Body);

                let prob = (
                  resultJSON[0].probability * 100
                ).toFixed(2);
                res.render("classify", { displayMessage, photoUrl, prob, imageClassification: resultJSON, source: "S3 Bucket" });
                // res.json({ source: "S3 Bucket", ...resultJSON });

                resultJSON.source = "Redis Cache";
                redisClient.setEx(persistenceKey, 3600, JSON.stringify(resultJSON));
              })
              .catch((err) => {
                if (err.statusCode === 404) {
                  /* Clasify the Image and store in Redis and S3 */
                  classifyImage(photoUrl)
                    .then((imageClassification) => {
                      const responseJSON = imageClassification;

                      /* Store in S3 Bucket*/
                      const body = JSON.stringify(responseJSON);
                      const objectParams = {
                        Bucket: bucketName,
                        Key: persistenceKey,
                        Body: body,
                      };

                      /* Store in Redis*/       
                      redisClient.setEx(
                        persistenceKey,
                        3600,
                        JSON.stringify({ source: "Redis Cache", ...responseJSON })
                      );

                      s3.putObject(objectParams)
                      .promise()
                      .then(() => {
                        console.log(
                          `Successfully uploaded data to ${bucketName}/${persistenceKey}`
                        ); 
                      });

                      console.log(imageClassification);
                      let prob = (
                        imageClassification[0].probability * 100
                      ).toFixed(2);
                      res.render("classify", {
                        displayMessage,
                        photoUrl,
                        prob,
                        imageClassification,
                        source: "Web"
                      });
                      // res.status(200).send({
                      //   classification: imageClassification,
                      // });
                    })
                    .catch((err) => {
                      console.log(err);
                      res
                        .status(500)
                        .send(
                          "Something went wrong while fetching image from URL."
                        );
                    });
                }
              });
          }
        });
      } else {
        /* If the query is garbage i.e no photos exist */
        let displayMessage = `Please enter a valid query, I don't know what ${searchTerm} means`;
        res.render("classify", { displayMessage });
      }
    })
    .catch((err) => {
      let status = err.status;
      let errMessage = err.message;

      res.status(500).render("error", { errMessage, status });
    });
});

/* ------------------------ HELPER FUNCTIONS ------------------------ */
function createAPIOptions(query) {
  const options = {
    hostname: "api.pexels.com",
    path: `/v1/search?query=${query}&per_page=1`,
    method: "GET",
  };

  return options;
}

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
