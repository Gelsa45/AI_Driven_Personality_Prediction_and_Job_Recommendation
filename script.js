document.addEventListener("DOMContentLoaded", function () {
    document.getElementById("result-section").classList.add("hidden");
});

function predictPersonality() {
    let userInput = document.getElementById("userInput").value.trim();
    let errorMsg = document.getElementById("error-message");
    let resultSection = document.getElementById("result-section");
    let loadingSection = document.getElementById("loading");

    errorMsg.innerText = "";
    resultSection.classList.add("hidden");

    if (!userInput) {
        errorMsg.innerText = "⚠️ Please enter some text.";
        return;
    }

    loadingSection.classList.remove("hidden");

    // Debugging: Log the payload being sent
    console.log("Sending request with payload:", JSON.stringify({ text: userInput }));

    // Send a POST request to the /predict endpoint
    fetch("/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({ text: userInput }), // Ensure the payload matches Flask's expectation
    })
        .then((response) => {
            console.log("Received response status:", response.status); // Debugging
            if (!response.ok) {
                // If the response is not OK, parse the error message from the JSON response
                return response.json().then(errData => {
                    throw new Error(errData.error || `Server error: ${response.statusText}`);
                });
            }
            return response.json();
        })
        .then((data) => {
            console.log("Received data:", data); // Debugging
            loadingSection.classList.add("hidden");

            if (data.error) {
                // Handle server-side errors
                errorMsg.innerText = `⚠️ ${data.error}`;
            } else {
                // Redirect to the result page with query parameters
                window.location.href = `/result?personality=${encodeURIComponent(
                    data.personality
                )}&description=${encodeURIComponent(
                    data.description
                )}&recommended_jobs=${encodeURIComponent(
                    data.recommended_jobs.join(", ")
                )}`;
            }
        })
        .catch((error) => {
            console.error("Error:", error); // Debugging
            errorMsg.innerText = `⚠️ ${error.message}`; // Display the specific error message
            loadingSection.classList.add("hidden");
        });
}