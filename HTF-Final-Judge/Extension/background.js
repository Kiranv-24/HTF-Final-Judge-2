chrome.runtime.onMessage.addListener(function (request, sender, sendResponse) {
  console.log("Received message:", request);
  if (request.action === "detectPhishing") {
    processURL(request.url).then(result => {
      sendResponse({ result: result });
    }).catch(error => {
      console.error("Error processing URL:", error);
      sendResponse({ result: "Error occurred" });
    });
    return true; // keep the message channel open for async
  }
});

async function processURL(url) {
  console.log("Received message:");
  if (!url) {
    console.error("URL is empty");
    return "Invalid URL";
  }

  try {
    const response = await fetch('http://127.0.0.1:2002/check_phishing', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url })
    });

    const data = await response.json();
    return data.result || data.result_text || "unknown result";
  } catch (error) {
    console.error('Fetch error:', error);
    return 'Error occurred';
  }
}
