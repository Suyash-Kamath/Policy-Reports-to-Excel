async function processFile() {
    const companyName = document.getElementById('company-name').value.trim();
    const fileInput = document.getElementById('file-upload');
    const processBtn = document.getElementById('process-btn');
    const statusDiv = document.getElementById('status');
    const extractedTextDiv = document.getElementById('extracted-text');
    const parsedDataDiv = document.getElementById('parsed-data');
    const downloadLink = document.getElementById('download-link');

    if (!companyName) {
        statusDiv.innerHTML = '<span class="error">Please enter a company name.</span>';
        return;
    }
    if (!fileInput.files[0]) {
        statusDiv.innerHTML = '<span class="error">Please select a file.</span>';
        return;
    }

    statusDiv.textContent = `Processing ${fileInput.files[0].name}...`;
    processBtn.disabled = true;
    extractedTextDiv.style.display = 'none';
    parsedDataDiv.style.display = 'none';
    downloadLink.style.display = 'none';

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    formData.append('company_name', companyName);

    try {
        const response = await fetch('http://127.0.0.1:8000/process-file/', { // use backend URL in dev
            method: 'POST',
            body: formData
        });

        let result;
        try {
            result = await response.json();
        } catch (err) {
            const text = await response.text(); // fallback if response isn’t JSON
            statusDiv.innerHTML = `<span class="error">Server returned non-JSON response: ${text}</span>`;
            console.error("Non-JSON response:", text);
            return;
        }

        if (!response.ok) {
            statusDiv.innerHTML = `<span class="error">Error processing ${fileInput.files[0].name}: ${result.detail || "Unknown error"}</span>`;
            console.error('Backend error:', result);
            return;
        }

        // ✅ Success
        extractedTextDiv.innerHTML = `<strong>Extracted Text:</strong><br>${result.extracted_text || 'No extracted text available'}`;
        extractedTextDiv.style.display = 'block';

        parsedDataDiv.innerHTML = `<strong>Parsed Data:</strong><br><pre>${JSON.stringify(result.parsed_data || {}, null, 2)}</pre>`;
        parsedDataDiv.style.display = 'block';

        // Excel download
        const excelBlob = new Blob([Uint8Array.from(atob(result.excel_file), c => c.charCodeAt(0))], {
            type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        });
        const url = window.URL.createObjectURL(excelBlob);
        downloadLink.href = url;
        downloadLink.download = 'policy_data.xlsx';
        downloadLink.textContent = 'Download Excel File';
        downloadLink.style.display = 'inline-block';

        statusDiv.innerHTML = `<span class="success">Processing complete for ${fileInput.files[0].name}! Download the Excel file below.</span>`;
    } catch (error) {
        statusDiv.innerHTML = `<span class="error">Error: ${error.message}. Please check your network, API key, or file content.</span>`;
        console.error('Fetch error:', error);
    } finally {
        processBtn.disabled = false;
    }
}
