function checkPhishing(url) {
  fetch('http://localhost:2002/check_phishing', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json' // Set Content-Type header to application/json
    },
    body: JSON.stringify({ url: url })
  })
  .then(response => response.json())
  .then(data => {
    displayResult(data);
  })
  .catch(error => {
    console.error('Error:', error);
    displayError();
  });
}


