function processURL() {
  console.log("Processing URL:", window.location.href);
  let domain_input = window.location.href;

  chrome.runtime.sendMessage(
    {
      action: "detectPhishing",
      url: domain_input
    },
    function(response) {
      if (chrome.runtime.lastError) {
        console.error("Error sending message:", chrome.runtime.lastError);
        alert("Extension error: " + chrome.runtime.lastError.message);
      } else {
        alert("Phishing Check Result: " + response.result);
      }
    }
  );
}

processURL();
