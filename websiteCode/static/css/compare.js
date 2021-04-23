
fetch("/compare")
  .then(function (response) {
    return response.json();
  })
  .then(function (text) {
  
    console.log(text);
    
    document.getElementById('Postcode1').innerHTML = "Postcode 1 - " + text[0]['inputPostcode'];
    document.getElementById('Postcode2').innerHTML = "Postcode 2 - " + text[0]['inputPostcode2'];
    
    document.getElementById('p1NumberOfCrimes').innerHTML = "Number of crimes reported is " + text[0]['NumberInwardCrimes'];
    document.getElementById('p2NumberOfCrimes').innerHTML = "Number of crimes reported is " + text[0]['NumberInwardCrimes2'];
       
    document.getElementById('p1Mode').innerHTML = "Mode crime is " + text[0]['ModeCrimeInward'];
    document.getElementById('p2Mode').innerHTML = "Mode crime is " + text[0]['ModeCrimeInward2'];
    
    document.getElementById('Postcode1Outward').innerHTML = "Outward Postcode 1 - " + text[0]['inputPostcodeOutward'];
    document.getElementById('Postcode2Outward').innerHTML = "Outward Postcode 2 - " + text[0]['inputPostcodeOutward2'];
    
    document.getElementById('p1NumberOfCrimesOutward').innerHTML = "Number of crimes reported is " + text[0]['NumberOutwardCrimes'];
    document.getElementById('p2NumberOfCrimesOutward').innerHTML = "Number of crimes reported is " + text[0]['NumberOutwardCrimes2'];
    
    document.getElementById('p1ModeOutward').innerHTML = "Mode crime is " + text[0]['ModeCrime'];
    document.getElementById('p2ModeOutward').innerHTML = "Mode crime is " + text[0]['ModeCrime2'];
    
    document.getElementById('p1PredictedNumberOfCrimes').innerHTML = "Predicted number of crimes (next month) is " + text[0]['PredictedNumberOfCrimes'];
    document.getElementById('p2PredictedNumberOfCrimes').innerHTML = "Predicted number of crimes (next month) is " + text[0]['PredictedNumberOfCrimes2'];

    document.getElementById('p1Severity').innerHTML = "CMAP crime severity rating is " + text[0]['SeverityRating'];
    document.getElementById('p2Severity').innerHTML = "CMAP crime severity rating is " + text[0]['SeverityRating2'];

    document.getElementById('dataTime').innerHTML = "Data from " + text[0]['FirstMonth'] + " to " + text[0]['FirstYear'];
    document.getElementById('dataTime2').innerHTML = "Data from " + text[0]['SecondMonth'] + " to " + text[0]['SecondYear'];
    
    document.getElementById('p1CrimeIncreasingDecreasing').innerHTML = "Crime is " + text[0]['CrimeIncreasingDecreasing'];
    document.getElementById('p2CrimeIncreasingDecreasing').innerHTML = "Crime is " + text[0]['CrimeIncreasingDecreasing2'];


});


