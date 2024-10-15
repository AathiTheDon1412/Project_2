const inputText = document.getElementById('input-text');
const generateButton = document.getElementById('generate-button');
const resultContainer = document.getElementById('result-container');

generateButton.addEventListener('click', async () => {
    const textDescription = inputText.value.trim();
    if (textDescription) {
        try {
            const response = await fetch('/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ textDescription }),
            });
            const threeDModel = await response.json();
            const resultHTML = `
                <h2>Generated 3D Model:</h2>
                <img src="data:image/png;base64,${threeDModel}" alt="3D Model">
            `;
            resultContainer.innerHTML = resultHTML;
        } catch (error) {
            console.error(error);
            resultContainer.innerHTML = '<p>Error generating 3D model.</p>';
        }
    } else {
        resultContainer.innerHTML = '<p>Please enter a text description.</p>';
    }
});