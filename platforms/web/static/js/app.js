document.addEventListener('DOMContentLoaded', function() {

    const uploadArea = document.getElementById('upload-area');
    const imageUpload = document.getElementById('image-upload');
    const previewImage = document.getElementById('preview-image');
    const previewContainer = document.getElementById('preview-container');
    const removeImageBtn = document.getElementById('remove-image');
    const continueToDetectionBtn = document.getElementById('continue-to-detection');


    const methodCards = document.querySelectorAll('.method-card');
    const processImageBtn = document.getElementById('process-image');
    const backToUploadBtn = document.getElementById('back-to-upload');


    const resultImage = document.getElementById('result-image');
    const debugImage = document.getElementById('debug-image');
    const debugPlaceholder = document.getElementById('debug-placeholder');
    const identifyBtn = document.getElementById('identify-plant');
    const backToDetectionBtn = document.getElementById('back-to-detection');
    const startOverBtn = document.getElementById('start-over');


    const plantIcon = document.getElementById('plant-icon');
    const plantName = document.getElementById('plant-name');
    const confidenceValue = document.getElementById('confidence-value');
    const confidenceBar = document.getElementById('confidence-bar');
    const plantInfo = document.getElementById('plant-info');
    const careInfo = document.getElementById('care-info');


    const tabHeaders = document.querySelectorAll('.tab-header');
    const tabContents = document.querySelectorAll('.tab-content');


    const statusIcon = document.getElementById('status-icon');
    const statusText = document.getElementById('status-text');


    const stepElements = {
        1: document.getElementById('step-1'),
        2: document.getElementById('step-2'),
        3: document.getElementById('step-3'),
    };

    const stepPanels = {
        1: document.getElementById('upload-panel'),
        2: document.getElementById('detection-panel'),
        3: document.getElementById('results-panel'),
    };


    const alternativeMatches = [];
    document.querySelectorAll('.alt-match').forEach(match => {
        const nameEl = match.querySelector('.alt-name');
        const confidenceEl = match.querySelector('.confidence-badge');
        alternativeMatches.push({
            nameEl,
            confidenceEl
        });
    });


    let currentImage = null;
    let currentDetection = null;
    let currentStep = 1;
    let currentMethod = 'multi';




    uploadArea.addEventListener('click', () => {
        imageUpload.click();
    });

    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');

        if (e.dataTransfer.files.length) {
            handleImageFile(e.dataTransfer.files[0]);
        }
    });

    imageUpload.addEventListener('change', () => {
        if (imageUpload.files.length) {
            handleImageFile(imageUpload.files[0]);
        }
    });

    removeImageBtn.addEventListener('click', () => {
        resetUploadArea();
    });

    continueToDetectionBtn.addEventListener('click', () => {
        if (!currentImage) return;
        goToStep(2);
    });


    methodCards.forEach(card => {
        card.addEventListener('click', () => {
            methodCards.forEach(c => c.classList.remove('active'));
            card.classList.add('active');
            currentMethod = card.dataset.method;
            updateStatus(`${card.querySelector('h4').textContent} selected`, 'ℹ️', 'info');
        });
    });

    processImageBtn.addEventListener('click', () => {
        processImage(currentImage);
    });

    backToUploadBtn.addEventListener('click', () => {
        goToStep(1);
    });


    identifyBtn.addEventListener('click', () => {
        if (!currentDetection) {
            updateStatus('Please process an image first', '⚠️', 'warning');
            return;
        }

        identifyPlant();
    });

    backToDetectionBtn.addEventListener('click', () => {
        goToStep(2);
    });

    startOverBtn.addEventListener('click', () => {
        resetInterface();
        goToStep(1);
    });


    tabHeaders.forEach(header => {
        header.addEventListener('click', () => {
            const tabId = header.dataset.tab;


            tabHeaders.forEach(h => h.classList.remove('active'));
            header.classList.add('active');


            tabContents.forEach(content => content.classList.remove('active'));
            document.getElementById(`${tabId}-tab`).classList.add('active');
        });
    });




    function handleImageFile(file) {
        if (!file.type.match('image.*')) {
            updateStatus('Please select an image file', '⚠️', 'warning');
            return;
        }

        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            previewContainer.classList.remove('hidden');
            uploadArea.classList.add('hidden');
            updateStatus('Image loaded, ready to continue', '✅', 'success');
            currentImage = file;


            continueToDetectionBtn.classList.remove('disabled');
        };
        reader.readAsDataURL(file);
    }


    function resetUploadArea() {
        previewContainer.classList.add('hidden');
        uploadArea.classList.remove('hidden');
        imageUpload.value = '';
        currentImage = null;
        continueToDetectionBtn.classList.add('disabled');
        updateStatus('Upload an image to begin', 'ℹ️', 'info');
    }


    function processImage(file) {
        updateStatus('Processing image...', '⏳', 'info');


        const formData = new FormData();
        formData.append('image', file);
        formData.append('detection_method', currentMethod);


        showLoadingState(processImageBtn, true);


        fetch('/detect', {
                method: 'POST',
                body: formData
            })
            .then(response => {
                if (!response.ok) {
                    return response.json()
                        .then(data => {
                            throw new Error(data.error || `Server error: ${response.status}`);
                        })
                        .catch(err => {

                            if (err instanceof SyntaxError) {
                                throw new Error(`Detection request failed: ${response.status}`);
                            }
                            throw err;
                        });
                }
                return response.json();
            })
            .then(data => {
                showLoadingState(processImageBtn, false);

                if (data.error) {
                    updateStatus(data.error, '❌', 'error');
                    console.error('Detection error:', data.error);
                    return;
                }


                currentDetection = data;


                if (data.frame) {
                    resultImage.src = `data:image/jpeg;base64,${data.frame}`;
                } else {
                    resultImage.src = previewImage.src;
                }


                if (data.debug_frame) {
                    debugImage.src = `data:image/jpeg;base64,${data.debug_frame}`;
                    debugImage.classList.remove('hidden');
                    debugPlaceholder.classList.add('hidden');
                }

                if (data.detection) {
                    updateStatus('Plant detected!', '✅', 'success');

                    if (data.bbox) {

                    }


                    if (data.identification) {

                        updateIdentificationUI(data.identification);


                        identifyBtn.classList.add('hidden');
                    } else {

                        identifyBtn.classList.remove('hidden');


                        if (data.identification_error) {
                            updateStatus(`Detection success but identification failed: ${data.identification_error}`, '⚠️', 'warning');
                        }
                    }


                    goToStep(3);
                } else {
                    updateStatus('No plants detected in image', '⚠️', 'warning');

                    goToStep(3);
                }
            })
            .catch(error => {
                showLoadingState(processImageBtn, false);
                updateStatus(`Error: ${error.message}`, '❌', 'error');
                console.error('Detection error:', error);
            });
    }


    function identifyPlant() {
        updateStatus('Identifying plant...', '⏳', 'info');


        showLoadingState(identifyBtn, true);


        fetch('/identify', {
                method: 'POST'
            })
            .then(response => {
                if (!response.ok) {
                    return response.json()
                        .then(data => {
                            throw new Error(data.error || `Server error: ${response.status}`);
                        })
                        .catch(err => {

                            if (err instanceof SyntaxError) {
                                throw new Error(`Identification request failed: ${response.status}`);
                            }
                            throw err;
                        });
                }
                return response.json();
            })
            .then(data => {
                showLoadingState(identifyBtn, false);

                if (data.error) {
                    updateStatus(data.error, '❌', 'error');
                    console.error('Identification error:', data.error);
                    return;
                }


                updateIdentificationUI(data);
            })
            .catch(error => {
                showLoadingState(identifyBtn, false);
                updateStatus(`Error: ${error.message}`, '❌', 'error');
                console.error('Identification error:', error);
            });
    }


    function updateIdentificationUI(identification) {

        if (identification.status === 'success') {
            plantIcon.textContent = '🌿';
            plantIcon.className = 'plant-icon text-success';
        } else if (identification.status === 'low_confidence') {
            plantIcon.textContent = '🌱';
            plantIcon.className = 'plant-icon text-warning';
        } else {
            plantIcon.textContent = '❓';
            plantIcon.className = 'plant-icon text-info';
        }


        if (identification.status === 'success' || identification.status === 'low_confidence') {

            const speciesName = formatSpeciesName(identification.class_name || identification.class_id);

            if (identification.status === 'low_confidence') {
                plantName.textContent = `${speciesName} (low confidence)`;
                plantName.className = 'plant-name text-warning';
            } else {
                plantName.textContent = speciesName;
                plantName.className = 'plant-name text-success';
            }


            const confidencePercent = identification.confidence * 100;
            confidenceValue.textContent = `${confidencePercent.toFixed(1)}%`;
            confidenceBar.style.width = `${confidencePercent}%`;


            if (confidencePercent >= 80) {
                confidenceBar.className = 'progress-value bg-success';
            } else if (confidencePercent >= 60) {
                confidenceBar.className = 'progress-value';
            } else if (confidencePercent >= 40) {
                confidenceBar.className = 'progress-value bg-info';
            } else {
                confidenceBar.className = 'progress-value bg-warning';
            }


            const topPredictions = identification.top_predictions || [];
            alternativeMatches.forEach((match, i) => {
                if (i < topPredictions.length) {
                    const pred = topPredictions[i];
                    const probPercent = pred.probability * 100;
                    const speciesName = formatSpeciesName(pred.class_name || pred.class_id);

                    match.nameEl.textContent = speciesName;
                    match.confidenceEl.textContent = `${probPercent.toFixed(1)}%`;
                } else {
                    match.nameEl.textContent = '--';
                    match.confidenceEl.textContent = '--';
                }
            });


            const plantInfoData = identification.plant_info;
            updatePlantInfoTab(plantInfoData, speciesName, identification.status);


            updateCareInfoTab(plantInfoData);


            document.querySelector('.tab-header[data-tab="info"]').click();

            updateStatus('Plant identification complete!', '✅', 'success');
        } else {

            plantName.textContent = 'No plant detected';
            plantName.className = 'plant-name';


            confidenceValue.textContent = '0%';
            confidenceBar.style.width = '0%';
            confidenceBar.className = 'progress-value';


            alternativeMatches.forEach(match => {
                match.nameEl.textContent = '--';
                match.confidenceEl.textContent = '--';
            });


            if (identification.status === 'error') {
                plantInfo.innerHTML = `
                    <div class="error-message">
                        <p class="text-error font-bold">Error: ${identification.message || 'Unknown error'}</p>
                        <p class="mt-2">Please try again or check the system configuration.</p>
                    </div>
                `;
                careInfo.innerHTML = '<p class="placeholder-text">Care information unavailable</p>';
                updateStatus('Identification failed', '❌', 'error');
            } else {
                plantInfo.innerHTML = `
                    <div class="no-plant-message">
                        <p>No plant detected in the current image.</p>
                        <p class="mt-4 font-bold">Try the following:</p>
                        <ul>
                            <li>Make sure a plant is visible in the image</li>
                            <li>Try a different detection method</li>
                            <li>Upload a clearer image</li>
                            <li>Try different lighting conditions</li>
                        </ul>
                    </div>
                `;
                careInfo.innerHTML = '<p class="placeholder-text">Care information unavailable</p>';
                updateStatus('No plant detected', '⚠️', 'warning');
            }
        }
    }


    function updatePlantInfoTab(plantInfoData, speciesName, status) {
        plantInfo.innerHTML = '';

        if (plantInfoData) {

            let infoHTML = `<h3 class="plant-info-heading">Scientific Name</h3>`;
            infoHTML += `<p>${plantInfoData.scientific_name || 'Unknown'}</p>`;

            infoHTML += `<h3 class="plant-info-heading">Family</h3>`;
            infoHTML += `<p>${plantInfoData.family || 'Unknown'}</p>`;

            if (plantInfoData.description) {
                infoHTML += `<h3 class="plant-info-heading">Description</h3>`;
                infoHTML += `<p>${plantInfoData.description}</p>`;
            }

            if (plantInfoData.common_varieties && plantInfoData.common_varieties.length) {
                infoHTML += `<h3 class="plant-info-heading">Common Varieties</h3>`;
                infoHTML += `<ul>`;
                plantInfoData.common_varieties.forEach(variety => {
                    infoHTML += `<li>${variety}</li>`;
                });
                infoHTML += `</ul>`;
            }

            plantInfo.innerHTML = infoHTML;
        } else {

            const speciesParts = speciesName.split(' ');
            let infoHTML = '';

            if (speciesParts.length >= 2) {
                const genus = speciesParts[0];
                const species = speciesParts.slice(1).join(' ');

                infoHTML += `<h3 class="plant-info-heading">Basic Information</h3>`;
                infoHTML += `<p><strong>Scientific Name:</strong> ${speciesName}</p>`;
                infoHTML += `<p><strong>Genus:</strong> ${genus}</p>`;
                infoHTML += `<p><strong>Species:</strong> ${species}</p>`;
            }

            if (status === 'low_confidence') {
                infoHTML += `<h3 class="plant-info-heading">Low Confidence Warning</h3>`;
                infoHTML += `<p>The identification system has low confidence in this match.</p>`;
                infoHTML += `<p class="font-bold">Possible reasons:</p>`;
                infoHTML += `<ul>`;
                infoHTML += `<li>Image quality is too low</li>`;
                infoHTML += `<li>Plant is not fully visible</li>`;
                infoHTML += `<li>Species is not in the training dataset</li>`;
                infoHTML += `<li>Similar looking plants causing confusion</li>`;
                infoHTML += `</ul>`;
            } else {
                infoHTML += `<h3 class="plant-info-heading">Note</h3>`;
                infoHTML += `<p>No detailed information available for this plant species.</p>`;
                infoHTML += `<p>Consider researching more about it online.</p>`;
            }

            plantInfo.innerHTML = infoHTML;
        }
    }


    function updateCareInfoTab(plantInfoData) {
        careInfo.innerHTML = '';

        if (plantInfoData && plantInfoData.care) {
            let careHTML = `<h3 class="plant-info-heading">Care Guide</h3>`;
            careHTML += `<ul>`;

            for (const [careType, careInfo] of Object.entries(plantInfoData.care)) {
                careHTML += `<li><strong>${formatCareType(careType)}:</strong> ${careInfo}</li>`;
            }

            careHTML += `</ul>`;
            careHTML += `<p class="mt-4 text-info"><i class="fas fa-info-circle mr-2"></i> These are general care guidelines. Adjust based on your specific growing conditions.</p>`;

            careInfo.innerHTML = careHTML;
        } else {
            careInfo.innerHTML = `
                <div class="no-care-info">
                    <p>No specific care information is available for this plant.</p>
                    <h3 class="plant-info-heading">General Care Tips</h3>
                    <ul>
                        <li><strong>Watering:</strong> Check soil moisture before watering. Most plants prefer soil that dries slightly between waterings.</li>
                        <li><strong>Light:</strong> Observe your plant's response to current light conditions and adjust as needed.</li>
                        <li><strong>Soil:</strong> Most plants thrive in well-draining soil with appropriate nutrients.</li>
                        <li><strong>Temperature:</strong> Most houseplants prefer temperatures between 65-75°F (18-24°C).</li>
                    </ul>
                </div>
            `;
        }
    }


    function showLoadingState(button, isLoading) {
        const originalText = button.dataset.originalText || button.innerHTML;

        if (isLoading) {

            if (!button.dataset.originalText) {
                button.dataset.originalText = button.innerHTML;
            }

            button.innerHTML = '<div class="loading-spinner mr-2"></div> Processing...';
            button.disabled = true;
        } else {
            button.innerHTML = originalText;
            button.disabled = false;
        }
    }


    function goToStep(stepNumber) {

        currentStep = stepNumber;


        Object.values(stepPanels).forEach(panel => {
            panel.classList.add('hidden');
        });


        stepPanels[stepNumber].classList.remove('hidden');


        for (let i = 1; i <= 3; i++) {
            const stepElement = stepElements[i];
            const stepCircle = stepElement.querySelector('.step-circle');

            if (i < stepNumber) {

                stepCircle.classList.add('completed');
                stepCircle.classList.remove('active');
                stepElement.classList.remove('active');
            } else if (i === stepNumber) {

                stepCircle.classList.add('active');
                stepCircle.classList.remove('completed');
                stepElement.classList.add('active');
            } else {

                stepCircle.classList.remove('active', 'completed');
                stepElement.classList.remove('active');
            }
        }


        const stepLines = document.querySelectorAll('.step-line');
        stepLines.forEach((line, index) => {
            if (index < stepNumber - 1) {
                line.classList.add('active');
            } else {
                line.classList.remove('active');
            }
        });


        if (stepNumber === 1) {
            if (currentImage) {
                updateStatus('Image loaded, ready to continue', '✅', 'success');
            } else {
                updateStatus('Upload an image to begin', 'ℹ️', 'info');
            }
        } else if (stepNumber === 2) {
            updateStatus('Select a detection method and process your image', 'ℹ️', 'info');
        } else if (stepNumber === 3) {
            if (currentDetection && currentDetection.detection) {
                updateStatus('Plant detected! Click "Identify Plant" to analyze', '✅', 'success');
            } else {
                updateStatus('No plants detected. You can try a different method or image', '⚠️', 'warning');
            }
        }
    }


    function resetInterface() {

        resetUploadArea();


        methodCards.forEach((card, index) => {
            if (index === 0) {
                card.classList.add('active');
            } else {
                card.classList.remove('active');
            }
        });
        currentMethod = 'multi';


        debugImage.classList.add('hidden');
        debugPlaceholder.classList.remove('hidden');

        plantIcon.textContent = '🌱';
        plantIcon.className = 'plant-icon';
        plantName.textContent = 'Plant Name';
        plantName.className = 'plant-name';
        confidenceValue.textContent = '0%';
        confidenceBar.style.width = '0%';
        confidenceBar.className = 'progress-value';

        plantInfo.innerHTML = '<p class="placeholder-text">Plant information will appear here after identification</p>';
        careInfo.innerHTML = '<p class="placeholder-text">Care information will appear here after identification</p>';


        alternativeMatches.forEach(match => {
            match.nameEl.textContent = '--';
            match.confidenceEl.textContent = '--';
        });


        currentImage = null;
        currentDetection = null;


        updateStatus('Upload an image to begin', 'ℹ️', 'info');
    }


    function updateStatus(message, icon, type) {
        statusText.textContent = message;
        statusIcon.textContent = icon;


        const statusBar = document.querySelector('.status-bar');
        statusBar.className = 'status-bar';

        if (type === 'success') {
            statusIcon.className = 'status-icon text-success';
        } else if (type === 'warning') {
            statusIcon.className = 'status-icon text-warning';
        } else if (type === 'error') {
            statusIcon.className = 'status-icon text-error';
        } else {
            statusIcon.className = 'status-icon text-info';
        }
    }


    function formatSpeciesName(name) {
        if (typeof name === 'number' || !isNaN(parseInt(name))) {
            return `Plant ${name}`;
        }


        return name.replace(/_/g, ' ')
            .replace(/\b\w/g, c => c.toUpperCase());
    }


    function formatCareType(type) {
        return type.replace(/_/g, ' ')
            .replace(/\b\w/g, c => c.toUpperCase());
    }


    function init() {
        resetInterface();
        goToStep(1);
    }


    init();
});