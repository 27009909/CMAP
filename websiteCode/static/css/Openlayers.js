var latitudeValue = 0.0;
var longitudeValue = 0.0;
var crimeTextMarker = "";

fetch("/predict")
  .then(function (response) {
    return response.json();
  })
  .then(function (text) {
    console.log(text);
    document.getElementById('SeverityLabel').innerHTML = text[0]['SeverityRating'];
    document.getElementById('ActualNumberOfCrimesLabelFullPostcode').innerHTML = text[0]['NumberInwardCrimes'];
    document.getElementById('ModeCrimeLabelFullPostcode').innerHTML = text[0]['ModeCrimeInward'];
    document.getElementById('NumberOfCrimesLabel').innerHTML = text[0]['PredictedNumberOfCrimes'];
    document.getElementById('ModeCrimeLabel').innerHTML = text[0]['ModeCrime'];
    
    document.getElementById('IncreasingDecreasingLabel').innerHTML = text[0]['CrimeIncreasingDecreasing'];
    
    document.getElementById('ActualNumberOfCrimesLabel').innerHTML = text[0]['NumberOutwardCrimes'];
    document.getElementById('outwardPostcode').innerHTML = text[0]['Outward Postcode'];
    document.getElementById('Postcode').innerHTML = text[0]['inputPostcode'];
    
    document.getElementById('DateFrom').innerHTML = "Data from " + text[0]['FirstMonth'] + " to " + text[0]['SecondMonth'];

    console.log("GET response:");
    console.log(text[0].MAPLatitude);
    console.log(text[0].MAPLongitude);
    infoText = text;

    var straitSource;
    var map;

    var dataF = ["0"];

    var alltimecrimestrHTML = "All crime ";
    var antisocialcrimestrHTML = "Anti-social behaviour ";
    var bicycletheftcrimestrHTML = "Bicycle theft ";
    var burglarycrimestrHTML = "Burglary ";
    var criminaldamageandarsoncrimestrHTML = "Criminal damage and arson ";
    var drugscrimestrHTML = "Drugs ";
    var othertheftcrimestrHTML = "Other theft ";
    var possessionofweaponscrimestrHTML = "Possession of weapons ";
    var publicordercrimestrHTML = "Public order ";
    var robberycrimestrHTML = "Robbery ";
    var shopliftingcrimestrHTML = "Shoplifting ";
    var theftfromthepersonstrHTML = "Theft from the person ";
    var vehiclecrimestrHTML = "Vehicle crime ";
    var violenceandsexualoffencescrimestrHTML = "Violence and sexual offences ";
    var othercrimestrHTML = "Other crime ";

    content = document.getElementById("popup-content");
    var center = ol.proj.transform(
      [text[0].MAPLongitude, text[0].MAPLatitude],
      "EPSG:4326",
      "EPSG:3857"
    ); //initial position of map
    
    var view = new ol.View({
      center: center,
      zoom: 15,
    });

    //raster layer on map
    var OSMBaseLayer = new ol.layer.Tile({
      source: new ol.source.OSM(),
    });

    straitSource = new ol.source.Vector({ wrapX: true });
    var straitsLayer = new ol.layer.Vector({
      source: straitSource,
    });

    map = new ol.Map({
      layers: [OSMBaseLayer, straitsLayer],
      target: "map2",
      view: view,
      controls: [new ol.control.FullScreen(), new ol.control.Zoom()],
    });

    // Popup showing the position the user clicked
    var container = document.getElementById("popup");
    var popup = new ol.Overlay({
      element: container,
      autoPan: true,
      autoPanAnimation: {
        duration: 250,
      },
    });
    map.addOverlay(popup);

    /* Add a pointermove handler to the map to render the popup.*/
    map.on("pointermove", function (evt) {
      var feature = map.forEachFeatureAtPixel(
        evt.pixel,
        function (feat, layer) {
          return feat;
        }
      );

      if (feature && feature.get("type") == "Point") {
        var coordinate = evt.coordinate; //default projection is EPSG:3857 you may want to use ol.proj.transform

        content.innerHTML = feature.get("desc");
        popup.setPosition(coordinate);
      } else {
        popup.setPosition(undefined);
      }
    });

    function onlyUnique(value, index, self) {
      return self.indexOf(value) === index;
    }

    function extractColumn(arr, column) {
      return arr.map((x) => x[column]);
    }

    function addPointGeom(data) {
      var uniquesPostcode = extractColumn(data, ["Postcode"]);
      console.log(uniquesPostcode);
      uniquesPostcode = uniquesPostcode.filter(onlyUnique);
      console.log(uniquesPostcode);
      var g = 0;
      for (g = 0; g < uniquesPostcode.length; g++) {
        //iterate through array...

        var DataTableFiltered = data.filter(
            (data) => data["Postcode"] == uniquesPostcode[g]
          ),
          longitude = DataTableFiltered[0].Longitude,
          latitude = DataTableFiltered[0].Latitude,
          postcodeVar = uniquesPostcode[g],
          monthVar = DataTableFiltered["Month Number"],
          yearVar = DataTableFiltered["Year Number"];
        //result = text.filter(text => text.Postcode == postcodeVar);

        var x;
        var j;
        var CrimeTextMarker = "";

        var alltimecrimestr = "";
        var antisocialcrimestr = "";
        var bicycletheftcrimestr = "";
        var burglarycrimestr = "";
        var criminaldamageandarsoncrimestr = "";
        var drugscrimestr = "";
        var othertheftcrimestr = "";
        var possessionofweaponscrimestr = "";
        var publicordercrimestr = "";
        var robberycrimestr = "";
        var shopliftingcrimestr = "";
        var theftfromthepersonstr = "";
        var vehiclecrimestr = "";
        var violenceandsexualoffencescrimestr = "";
        var othercrimestr = "";

        var alltimecrimedate = "";
        var antisocialcrimedate = "";
        var bicycletheftcrimedate = "";
        var burglarycrimedate = "";
        var criminaldamageandarsoncrimedate = "";
        var drugscrimedate = "";
        var othertheftcrimedate = "";
        var possessionofweaponscrimedate = "";
        var publicordercrimedate = "";
        var robberycrimedate = "";
        var shopliftingcrimedate = "";
        var theftfromthepersondate = "";
        var vehiclecrimedate = "";
        var violenceandsexualoffencescrimedate = "";
        var othercrimedate = "";

        var alltimecrimeint = 0;
        var antisocialcrimeint = 0;
        var bicycletheftcrimeint = 0;
        var burglarycrimeint = 0;
        var criminaldamageandarsoncrimeint = 0;
        var drugscrimeint = 0;
        var othertheftcrimeint = 0;
        var possessionofweaponscrimeint = 0;
        var publicordercrimeint = 0;
        var robberycrimeint = 0;
        var shopliftingcrimeint = 0;
        var theftfromthepersonint = 0;
        var vehiclecrimeint = 0;
        var violenceandsexualoffencescrimeint = 0;
        var othercrimeint = 0;

        var varalltimecrimeint = 0;
        var varantisocialcrimeint = 0;
        var varbicycletheftcrimeint = 0;
        var varburglarycrimeint = 0;
        var varcriminaldamageandarsoncrimeint = 0;
        var vardrugscrimeint = 0;
        var varothertheftcrimeint = 0;
        var varpossessionofweaponscrimeint = 0;
        var varpublicordercrimeint = 0;
        var varrobberycrimeint = 0;
        var varshopliftingcrimeint = 0;
        var vartheftfromthepersonint = 0;
        var varvehiclecrimeint = 0;
        var varviolenceandsexualoffencescrimeint = 0;
        var varothercrimeint = 0;

        var n = dataF.includes(postcodeVar);
        //console.log('n is '  + n)
        if (n == false) {
          dataF = [...dataF, postcodeVar];
          //console.log('Lol' + data.toString())
          var duplicates = [];

          for (j = 0; j < DataTableFiltered.length; j++) {
            duplicates = [...duplicates, DataTableFiltered[j]["Month in Unix"]];
          }
          //console.log('DUPLICATES + ' + (duplicates).toString());

          var uniques = duplicates.filter(onlyUnique);
          //console.log('RESULT + ' + (uniques).toString() + ' ' + postcodeVar);

          for (j = 0; j < uniques.length; j++) {
            crimesInFilterMonth = DataTableFiltered.filter(
              (DataTableFiltered) =>
                DataTableFiltered["Month in Unix"] == uniques[j]
            );
            //Crimedate = (crimesInFilterMonth[j]['Month Number']).toString() + "/" + (crimesInFilterMonth[j]['Year Number']).toString()

            var s = new Date(uniques[j] * 1000).toLocaleDateString("en-GB");
            //console.log(s)

            Crimedate = s.slice(3, 8);
            for (x = 0; x < crimesInFilterMonth.length; x++) {
              //CrimeTextMarker += crimesInFilterMonth[x]['Crime type'] + "   "+ (crimesInFilterMonth[x]['Month Number']).toString() + "/" + (crimesInFilterMonth[x]['Year Number']).toString()+"\n";

              if (crimesInFilterMonth[x]["Crime type"] == "All crime") {
                NumbercrimesInFilterMonth = crimesInFilterMonth.filter(
                  (crimesInFilterMonth) =>
                    crimesInFilterMonth["Crime type"] == "All crime"
                );
                alltimecrimeint = NumbercrimesInFilterMonth.length;
                varalltimecrimeint += 1;

                if (varalltimecrimeint == 1) {
                  alltimecrimestr +=
                    " (" +
                    alltimecrimeint +
                    ")" +
                    "-" +
                    NumbercrimesInFilterMonth[0]["Month Number"].toString() +
                    "/" +
                    NumbercrimesInFilterMonth[0]["Year Number"].toString();
                }
              }
              if (
                crimesInFilterMonth[x]["Crime type"] == "Anti-social behaviour"
              ) {
                NumbercrimesInFilterMonth = crimesInFilterMonth.filter(
                  (crimesInFilterMonth) =>
                    crimesInFilterMonth["Crime type"] == "Anti-social behaviour"
                );
                antisocialcrimeint = NumbercrimesInFilterMonth.length;
                varantisocialcrimeint += 1;

                if (varantisocialcrimeint == 1) {
                  antisocialcrimestr +=
                    " (" +
                    antisocialcrimeint +
                    ")" +
                    "-" +
                    NumbercrimesInFilterMonth[0]["Month Number"].toString() +
                    "/" +
                    NumbercrimesInFilterMonth[0]["Year Number"].toString();
                }
              }
              if (crimesInFilterMonth[x]["Crime type"] == "Bicycle theft") {
                NumbercrimesInFilterMonth = crimesInFilterMonth.filter(
                  (crimesInFilterMonth) =>
                    crimesInFilterMonth["Crime type"] == "Bicycle theft"
                );
                bicycletheftcrimeint = NumbercrimesInFilterMonth.length;

                varbicycletheftcrimeint += 1;

                if (varbicycletheftcrimeint == 1) {
                  bicycletheftcrimestr +=
                    " (" +
                    bicycletheftcrimeint +
                    ")" +
                    "-" +
                    NumbercrimesInFilterMonth[0]["Month Number"].toString() +
                    "/" +
                    NumbercrimesInFilterMonth[0]["Year Number"].toString();
                }
              }
              if (crimesInFilterMonth[x]["Crime type"] == "Burglary") {
                NumbercrimesInFilterMonth = crimesInFilterMonth.filter(
                  (crimesInFilterMonth) =>
                    crimesInFilterMonth["Crime type"] == "Burglary"
                );
                burglarycrimeint = NumbercrimesInFilterMonth.length;

                varburglarycrimeint += 1;

                if (varburglarycrimeint == 1) {
                  burglarycrimestr +=
                    " (" +
                    burglarycrimeint +
                    ")" +
                    "-" +
                    NumbercrimesInFilterMonth[0]["Month Number"].toString() +
                    "/" +
                    NumbercrimesInFilterMonth[0]["Year Number"].toString();
                }
              }
              if (
                crimesInFilterMonth[x]["Crime type"] ==
                "Criminal damage and arson"
              ) {
                NumbercrimesInFilterMonth = crimesInFilterMonth.filter(
                  (crimesInFilterMonth) =>
                    crimesInFilterMonth["Crime type"] ==
                    "Criminal damage and arson"
                );
                criminaldamageandarsoncrimeint =
                  NumbercrimesInFilterMonth.length;

                varcriminaldamageandarsoncrimeint += 1;
                if (varcriminaldamageandarsoncrimeint == 1) {
                  criminaldamageandarsoncrimestr +=
                    " (" +
                    criminaldamageandarsoncrimeint +
                    ")" +
                    "-" +
                    NumbercrimesInFilterMonth[0]["Month Number"].toString() +
                    "/" +
                    NumbercrimesInFilterMonth[0]["Year Number"].toString();
                }
              }
              if (crimesInFilterMonth[x]["Crime type"] == "Drugs") {
                NumbercrimesInFilterMonth = crimesInFilterMonth.filter(
                  (crimesInFilterMonth) =>
                    crimesInFilterMonth["Crime type"] == "Drugs"
                );
                drugscrimeint = NumbercrimesInFilterMonth.length;

                vardrugscrimeint += 1;
                if (vardrugscrimeint == 1) {
                  drugscrimestr +=
                    " (" +
                    drugscrimeint +
                    ")" +
                    "-" +
                    NumbercrimesInFilterMonth[0]["Month Number"].toString() +
                    "/" +
                    NumbercrimesInFilterMonth[0]["Year Number"].toString();
                }
              }
              if (crimesInFilterMonth[x]["Crime type"] == "Other theft") {
                NumbercrimesInFilterMonth = crimesInFilterMonth.filter(
                  (crimesInFilterMonth) =>
                    crimesInFilterMonth["Crime type"] == "Other theft"
                );
                othertheftcrimeint = NumbercrimesInFilterMonth.length;

                varothertheftcrimeint += 1;
                if (varothertheftcrimeint == 1) {
                  othertheftcrimestr +=
                    " (" +
                    othertheftcrimeint +
                    ")" +
                    "-" +
                    NumbercrimesInFilterMonth[0]["Month Number"].toString() +
                    "/" +
                    NumbercrimesInFilterMonth[0]["Year Number"].toString();
                }
              }
              if (
                crimesInFilterMonth[x]["Crime type"] == "Possession of weapons"
              ) {
                NumbercrimesInFilterMonth = crimesInFilterMonth.filter(
                  (crimesInFilterMonth) =>
                    crimesInFilterMonth["Crime type"] == "Possession of weapons"
                );
                possessionofweaponscrimeint = NumbercrimesInFilterMonth.length;
                varpossessionofweaponscrimeint += 1;
                if (varpossessionofweaponscrimeint == 1) {
                  possessionofweaponscrimestr +=
                    " (" +
                    possessionofweaponscrimeint +
                    ")" +
                    "-" +
                    NumbercrimesInFilterMonth[0]["Month Number"].toString() +
                    "/" +
                    NumbercrimesInFilterMonth[0]["Year Number"].toString();
                }
              }
              if (crimesInFilterMonth[x]["Crime type"] == "Public order") {
                NumbercrimesInFilterMonth = crimesInFilterMonth.filter(
                  (crimesInFilterMonth) =>
                    crimesInFilterMonth["Crime type"] == "Public order"
                );
                publicordercrimeint = NumbercrimesInFilterMonth.length;
                varpublicordercrimeint += 1;
                if (varpublicordercrimeint == 1) {
                  publicordercrimestr +=
                    " (" +
                    publicordercrimeint +
                    ")" +
                    "-" +
                    NumbercrimesInFilterMonth[0]["Month Number"].toString() +
                    "/" +
                    NumbercrimesInFilterMonth[0]["Year Number"].toString();
                }
              }
              if (crimesInFilterMonth[x]["Crime type"] == "Robbery") {
                NumbercrimesInFilterMonth = crimesInFilterMonth.filter(
                  (crimesInFilterMonth) =>
                    crimesInFilterMonth["Crime type"] == "Robbery"
                );
                robberycrimeint = NumbercrimesInFilterMonth.length;
                varrobberycrimeint += 1;
                if (varrobberycrimeint == 1) {
                  robberycrimestr +=
                    " (" +
                    robberycrimeint +
                    ")" +
                    "-" +
                    NumbercrimesInFilterMonth[0]["Month Number"].toString() +
                    "/" +
                    NumbercrimesInFilterMonth[0]["Year Number"].toString();
                }
              }
              if (crimesInFilterMonth[x]["Crime type"] == "Shoplifting") {
                NumbercrimesInFilterMonth = crimesInFilterMonth.filter(
                  (crimesInFilterMonth) =>
                    crimesInFilterMonth["Crime type"] == "Shoplifting"
                );
                shopliftingcrimeint = NumbercrimesInFilterMonth.length;

                varshopliftingcrimeint += 1;
                if (varshopliftingcrimeint == 1) {
                  shopliftingcrimestr +=
                    " (" +
                    shopliftingcrimeint +
                    ")" +
                    "-" +
                    NumbercrimesInFilterMonth[0]["Month Number"].toString() +
                    "/" +
                    NumbercrimesInFilterMonth[0]["Year Number"].toString();
                }
              }
              if (
                crimesInFilterMonth[x]["Crime type"] == "Theft from the person"
              ) {
                NumbercrimesInFilterMonth = crimesInFilterMonth.filter(
                  (crimesInFilterMonth) =>
                    crimesInFilterMonth["Crime type"] == "Theft from the person"
                );
                theftfromthepersonint = NumbercrimesInFilterMonth.length;

                vartheftfromthepersonint += 1;
                if (vartheftfromthepersonint == 1) {
                  theftfromthepersonstr +=
                    " (" +
                    theftfromthepersonint +
                    ")" +
                    "-" +
                    NumbercrimesInFilterMonth[0]["Month Number"].toString() +
                    "/" +
                    NumbercrimesInFilterMonth[0]["Year Number"].toString();
                }
              }
              if (crimesInFilterMonth[x]["Crime type"] == "Vehicle crime") {
                NumbercrimesInFilterMonth = crimesInFilterMonth.filter(
                  (crimesInFilterMonth) =>
                    crimesInFilterMonth["Crime type"] == "Vehicle crime"
                );
                vehiclecrimeint = NumbercrimesInFilterMonth.length;

                varvehiclecrimeint += 1;
                if (varvehiclecrimeint == 1) {
                  vehiclecrimestr +=
                    " (" +
                    vehiclecrimeint +
                    ")" +
                    "-" +
                    NumbercrimesInFilterMonth[0]["Month Number"].toString() +
                    "/" +
                    NumbercrimesInFilterMonth[0]["Year Number"].toString();
                }
              }
              if (
                crimesInFilterMonth[x]["Crime type"] ==
                "Violence and sexual offences"
              ) {
                NumbercrimesInFilterMonth = crimesInFilterMonth.filter(
                  (crimesInFilterMonth) =>
                    crimesInFilterMonth["Crime type"] ==
                    "Violence and sexual offences"
                );
                violenceandsexualoffencescrimeint =
                  NumbercrimesInFilterMonth.length;
                varviolenceandsexualoffencescrimeint += 1;
                if (varviolenceandsexualoffencescrimeint == 1) {
                  violenceandsexualoffencescrimestr +=
                    " (" +
                    violenceandsexualoffencescrimeint +
                    ")" +
                    "-" +
                    NumbercrimesInFilterMonth[0]["Month Number"].toString() +
                    "/" +
                    NumbercrimesInFilterMonth[0]["Year Number"].toString();
                }
              }
              if (crimesInFilterMonth[x]["Crime type"] == "Other crime") {
                NumbercrimesInFilterMonth = crimesInFilterMonth.filter(
                  (crimesInFilterMonth) =>
                    crimesInFilterMonth["Crime type"] == "Other crime"
                );
                othercrimeint = NumbercrimesInFilterMonth.length;
                varothercrimeint += 1;
                if (varothercrimeint == 1) {
                  othercrimestr +=
                    " (" +
                    othercrimeint +
                    ")" +
                    "-" +
                    NumbercrimesInFilterMonth[0]["Month Number"].toString() +
                    "/" +
                    NumbercrimesInFilterMonth[0]["Year Number"].toString();
                }
              }
            }

            alltimecrimeint = 0;
            antisocialcrimeint = 0;
            bicycletheftcrimeint = 0;
            burglarycrimeint = 0;
            criminaldamageandarsoncrimeint = 0;
            drugscrimeint = 0;
            othertheftcrimeint = 0;
            possessionofweaponscrimeint = 0;
            publicordercrimeint = 0;
            robberycrimeint = 0;
            shopliftingcrimeint = 0;
            theftfromthepersonint = 0;
            vehiclecrimeint = 0;
            violenceandsexualoffencescrimeint = 0;
            othercrimeint = 0;

            varalltimecrimeint = 0;
            varantisocialcrimeint = 0;
            varbicycletheftcrimeint = 0;
            varburglarycrimeint = 0;
            varcriminaldamageandarsoncrimeint = 0;
            vardrugscrimeint = 0;
            varothertheftcrimeint = 0;
            varpossessionofweaponscrimeint = 0;
            varpublicordercrimeint = 0;
            varrobberycrimeint = 0;
            varshopliftingcrimeint = 0;
            vartheftfromthepersonint = 0;
            varvehiclecrimeint = 0;
            varviolenceandsexualoffencescrimeint = 0;
            varothercrimeint = 0;
          }

          alltimecrimestrHTML = alltimecrimestrHTML + " " + alltimecrimestr;
          antisocialcrimestrHTML =
            antisocialcrimestrHTML + " " + antisocialcrimestr;
          bicycletheftcrimestrHTML =
            bicycletheftcrimestrHTML + " " + bicycletheftcrimestr;
          burglarycrimestrHTML = burglarycrimestrHTML + " " + burglarycrimestr;
          criminaldamageandarsoncrimestrHTML =
            criminaldamageandarsoncrimestrHTML +
            " " +
            criminaldamageandarsoncrimestr;
          drugscrimestrHTML = drugscrimestrHTML + " " + drugscrimestr;
          othertheftcrimestrHTML =
            othertheftcrimestrHTML + " " + othertheftcrimestr;
          possessionofweaponscrimestrHTML =
            possessionofweaponscrimestrHTML + " " + possessionofweaponscrimestr;
          publicordercrimestrHTML =
            publicordercrimestrHTML + " " + publicordercrimestr;
          robberycrimestrHTML = robberycrimestrHTML + " " + robberycrimestr;
          shopliftingcrimestrHTML =
            shopliftingcrimestrHTML + " " + shopliftingcrimestr;
          theftfromthepersonstrHTML =
            theftfromthepersonstrHTML + " " + theftfromthepersonstr;
          vehiclecrimestrHTML = vehiclecrimestrHTML + " " + vehiclecrimestr;
          violenceandsexualoffencescrimestrHTML =
            violenceandsexualoffencescrimestrHTML +
            " " +
            violenceandsexualoffencescrimestr;
          othercrimestrHTML = othercrimestrHTML + " " + othercrimestr;

          //console.log(postcodeVar + alltimecrimestrHTML + antisocialcrimestrHTML + bicycletheftcrimestrHTML + burglarycrimestrHTML + criminaldamageandarsoncrimestrHTML + drugscrimestrHTML + othertheftcrimestrHTML + possessionofweaponscrimestrHTML + publicordercrimestrHTML + robberycrimestrHTML + shopliftingcrimestrHTML + theftfromthepersonstrHTML + vehiclecrimestrHTML + violenceandsexualoffencescrimestrHTML + othercrimestrHTML);
          var crimeTextForMarker = "";

          if (alltimecrimestrHTML != "All crime  ") {
            crimeTextForMarker += alltimecrimestrHTML + " \n ";
          }
          if (antisocialcrimestrHTML != "Anti-social behaviour  ") {
            crimeTextForMarker += antisocialcrimestrHTML + " \n ";
          }
          if (bicycletheftcrimestrHTML != "Bicycle theft  ") {
            crimeTextForMarker += bicycletheftcrimestrHTML + " \n ";
          }
          if (burglarycrimestrHTML != "Burglary  ") {
            crimeTextForMarker += burglarycrimestrHTML + " \n ";
          }
          if (
            criminaldamageandarsoncrimestrHTML != "Criminal damage and arson  "
          ) {
            crimeTextForMarker += criminaldamageandarsoncrimestrHTML + " \n ";
          }
          if (drugscrimestrHTML != "Drugs  ") {
            crimeTextForMarker += drugscrimestrHTML + " \n ";
          }
          if (othertheftcrimestrHTML != "Other theft  ") {
            crimeTextForMarker += othertheftcrimestrHTML + " \n ";
          }
          if (possessionofweaponscrimestrHTML != "Possession of weapons  ") {
            crimeTextForMarker += possessionofweaponscrimestrHTML + " \n ";
          }
          if (publicordercrimestrHTML != "Public order  ") {
            crimeTextForMarker += publicordercrimestrHTML + " \n ";
          }
          if (robberycrimestrHTML != "Robbery  ") {
            crimeTextForMarker += robberycrimestrHTML + " \n ";
          }
          if (shopliftingcrimestrHTML != "Shoplifting  ") {
            crimeTextForMarker += shopliftingcrimestrHTML + " \n ";
          }
          if (theftfromthepersonstrHTML != "Theft from the person  ") {
            crimeTextForMarker += theftfromthepersonstrHTML + " \n ";
          }
          if (vehiclecrimestrHTML != "Vehicle crime  ") {
            crimeTextForMarker += vehiclecrimestrHTML + " \n ";
          }
          if (
            violenceandsexualoffencescrimestrHTML !=
            "Violence and sexual offences  "
          ) {
            crimeTextForMarker += violenceandsexualoffencescrimestrHTML + " \n ";
          }
          if (othercrimestrHTML != "Other crime  ") {
            crimeTextForMarker += othercrimestrHTML + " \n ";
          }

          //console.log('LOL ' + postcodeVar + " " + crimeTextForMarker);
          (iconFeature = new ol.Feature({
            geometry: new ol.geom.Point(
              ol.proj.transform([longitude, latitude], "EPSG:4326", "EPSG:3857")
            ),
            type: "Point",
            desc:
              "<h2>" +
              "Postcode : " +
              postcodeVar + "</h2> \n\n" +
              "<h3>Crimes reported :" + "</h3>" + "<pre>" +
              "<p>" +
              crimeTextForMarker +
              "</p>" + "</pre>",
          })),
            (iconStyle = new ol.style.Style({
              image: new ol.style.Circle({
                radius: 5,
                stroke: new ol.style.Stroke({
                  color: [7, 65, 173, 1],
                }),
                fill: new ol.style.Fill({
                  color: [7, 65, 173, 0.8],
                }),
              }),
            }));

          iconFeature.setStyle(iconStyle);

          straitSource.addFeature(iconFeature);

          alltimecrimestrHTML = "All crime ";
          antisocialcrimestrHTML = "Anti-social behaviour ";
          bicycletheftcrimestrHTML = "Bicycle theft ";
          burglarycrimestrHTML = "Burglary ";
          criminaldamageandarsoncrimestrHTML = "Criminal damage and arson ";
          drugscrimestrHTML = "Drugs ";
          othertheftcrimestrHTML = "Other theft ";
          possessionofweaponscrimestrHTML = "Possession of weapons ";
          publicordercrimestrHTML = "Public order ";
          robberycrimestrHTML = "Robbery ";
          shopliftingcrimestrHTML = "Shoplifting ";
          theftfromthepersonstrHTML = "Theft from the person ";
          vehiclecrimestrHTML = "Vehicle crime ";
          violenceandsexualoffencescrimestrHTML =
            "Violence and sexual offences ";
          othercrimestrHTML = "Other crime ";
        }
      }
      //console.log(uniquesPostcode);
    } // End of function showStraits()

    addPointGeom(text);
  });
