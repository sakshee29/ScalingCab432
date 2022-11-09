var express = require('express');
const sharp = require("sharp");
const mobilenet = require('@tensorflow-models/mobilenet');
const fs = require("fs");
const formidable = require("formidable");
const bodyParser = require("body-parser");



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

module.exports = router;
