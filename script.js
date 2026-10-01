/**
 * ==============================================================================
 * Agricultural AI - Satellite Image Upload & Inspection Frontend
 * ==============================================================================
 * This script handles:
 *  1. Drag-and-drop & file selection events (.tif, .tiff, .png, .jpg, .jpeg)
 *  2. File metadata extraction
 *  3. Multispectral GeoTIFF backend processing via FastAPI (/api/analyze)
 *  4. Real satellite band value extraction (B02 Blue, B03 Green, B04 Red, B08 NIR)
 *  5. Calculating 6 core bio-optical and canopy features:
 *     NDVI, EVI, FVC, LAI, NDVI Texture, Biomass
 *  6. Passing exact features to pre-trained Random Forest models to predict N, P, K
 *  7. Realistic ~3s non-blocking loading progression through the 5 analysis stages
 *  8. Displaying Section 1 (Satellite Analysis), Section 2 (Nutrient Estimation),
 *     and Section 3 (Groq AI Agricultural Advisor) with scientific disclaimers
 * ==============================================================================
 */

document.addEventListener("DOMContentLoaded", () => {
  // --------------------------------------------------------------------------
  // 1. DOM Element References
  // --------------------------------------------------------------------------
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("fileInput");
  const browseBtn = document.getElementById("browseBtn");
  const dropzoneContent = document.getElementById("dropzoneContent");
  const uploadError = document.getElementById("uploadError");
  const errorMessage = document.getElementById("errorMessage");

  // Crop & Growth Stage Controls
  const cropSelect = document.getElementById("cropSelect");
  const growthStageSelect = document.getElementById("growthStageSelect");

  // Selected file preview elements
  const filePreviewCard = document.getElementById("filePreviewCard");
  const fileBadgeIcon = document.getElementById("fileBadgeIcon");
  const previewFileName = document.getElementById("previewFileName");
  const previewFileSize = document.getElementById("previewFileSize");
  const previewFileType = document.getElementById("previewFileType");
  const removeFileBtn = document.getElementById("removeFileBtn");
  const imagePreviewElement = document.getElementById("imagePreviewElement");
  const geotiffNotice = document.getElementById("geotiffNotice");

  // Retrieved Satellite Data (Metadata) Table elements
  const metadataStatusIndicator = document.getElementById("metadataStatusIndicator");
  const metaFileName = document.getElementById("metaFileName");
  const metaFileType = document.getElementById("metaFileType");
  const metaFileSize = document.getElementById("metaFileSize");
  const metaImageWidth = document.getElementById("metaImageWidth");
  const metaImageHeight = document.getElementById("metaImageHeight");
  const metaNumberBands = document.getElementById("metaNumberBands");
  const metaSpatialRes = document.getElementById("metaSpatialRes");
  const metaCRS = document.getElementById("metaCRS");

  // Spectral Band Value display elements
  const bandValues = {
    blue: document.getElementById("bandBlueVal"),
    green: document.getElementById("bandGreenVal"),
    red: document.getElementById("bandRedVal"),
    redEdge: document.getElementById("bandRedEdgeVal"),
    nir: document.getElementById("bandNirVal"),
    swir: document.getElementById("bandSwirVal")
  };

  // Section 1: Satellite Analysis status elements
  const satelliteAnalysisBadge = document.getElementById("satelliteAnalysisBadge");
  const paramStatuses = {
    ndvi: document.getElementById("paramStatusNDVI"),
    evi: document.getElementById("paramStatusEVI"),
    fvc: document.getElementById("paramStatusFVC"),
    lai: document.getElementById("paramStatusLAI"),
    texture: document.getElementById("paramStatusTexture"),
    zone: document.getElementById("paramStatusZone"),
    biomass: document.getElementById("paramStatusBiomass"),
    lst: document.getElementById("paramStatusLST")
  };

  // Section 2: Nutrient Estimation Elements
  const nutrientStatusBadge = document.getElementById("nutrientStatusBadge");
  const predNVal = document.getElementById("predNVal");
  const predPVal = document.getElementById("predPVal");
  const predKVal = document.getElementById("predKVal");

  // Section 3: Groq AI Agricultural Advisor Elements
  const advisorStatusBadge = document.getElementById("advisorStatusBadge");
  const advisorEmptyState = document.getElementById("advisorEmptyState");
  const advisorResults = document.getElementById("advisorResults");
  const advisorSummary = document.getElementById("advisorSummary");
  const advisorObservationsList = document.getElementById("advisorObservationsList");
  const advisorConcernsList = document.getElementById("advisorConcernsList");
  const advisorActionsList = document.getElementById("advisorActionsList");
  const advisorMonitorList = document.getElementById("advisorMonitorList");
  const advisorValidationNote = document.getElementById("advisorValidationNote");
  const advisorFallbackBanner = document.getElementById("advisorFallbackBanner");
  const advisorFallbackMsg = document.getElementById("advisorFallbackMsg");

  // Analysis Progress Modal Elements
  const analysisOverlay = document.getElementById("analysisOverlay");
  const analysisCurrentStageText = document.getElementById("analysisCurrentStageText");
  const analysisProgressBar = document.getElementById("analysisProgressBar");
  const timelineSteps = {
    upload: document.getElementById("step-upload"),
    bands: document.getElementById("step-bands"),
    features: document.getElementById("step-features"),
    npk: document.getElementById("step-npk"),
    groq: document.getElementById("step-groq"),
    complete: document.getElementById("step-complete")
  };

  // Action Button & status elements
  const calcParamsBtn = document.getElementById("calcParamsBtn");
  const calcBtnText = document.getElementById("calcBtnText");
  const calcActionNote = document.getElementById("calcActionNote");
  const calcActionNoteText = document.getElementById("calcActionNoteText");
  const backendStatusBadge = document.getElementById("backendStatusBadge");

  // Current active file reference
  let currentActiveFile = null;

  // --------------------------------------------------------------------------
  // 2. Constants & Allowed Extensions
  // --------------------------------------------------------------------------
  const ALLOWED_EXTENSIONS = ["tif", "tiff", "png", "jpg", "jpeg"];
  const MIN_DISPLAY_MS = 3000;

  // --------------------------------------------------------------------------
  // 3. Helper Functions
  // --------------------------------------------------------------------------
  function formatBytes(bytes, decimals = 2) {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + " " + sizes[i];
  }

  function getFileExtension(filename) {
    return filename.slice(((filename.lastIndexOf(".") - 1) >>> 0) + 2).toLowerCase();
  }

  function isFileValid(file) {
    const ext = getFileExtension(file.name);
    return ALLOWED_EXTENSIONS.includes(ext);
  }

  function showError(msg) {
    errorMessage.textContent = msg;
    uploadError.classList.remove("hidden");
  }

  function clearError() {
    uploadError.classList.add("hidden");
    errorMessage.textContent = "";
  }

  // --------------------------------------------------------------------------
  // 4. File Handling & Preview Pipeline
  // --------------------------------------------------------------------------
  function processFile(file) {
    clearError();
    if (!file) return;

    if (!isFileValid(file)) {
      const ext = getFileExtension(file.name);
      showError(`Unsupported file format (.${ext || "unknown"}). Allowed formats: GeoTIFF (.tif, .tiff), PNG, JPG/JPEG.`);
      resetDashboard();
      return;
    }

    currentActiveFile = file;
    const ext = getFileExtension(file.name);
    const isTiff = ext === "tif" || ext === "tiff";
    const formattedSize = formatBytes(file.size);

    previewFileName.textContent = file.name;
    previewFileSize.textContent = formattedSize;
    previewFileType.textContent = isTiff ? "GEOTIFF" : ext.toUpperCase();

    if (isTiff) {
      fileBadgeIcon.innerHTML = `
        <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="2">
          <polygon points="12 2 2 7 12 12 22 7 12 2"></polygon>
          <polyline points="2 17 12 22 22 17"></polyline>
          <polyline points="2 12 12 17 22 12"></polyline>
        </svg>
      `;
      imagePreviewElement.classList.add("hidden");
      imagePreviewElement.src = "";
      geotiffNotice.classList.remove("hidden");
    } else {
      fileBadgeIcon.innerHTML = `
        <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="2">
          <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
          <circle cx="8.5" cy="8.5" r="1.5"></circle>
          <polyline points="21 15 16 10 5 21"></polyline>
        </svg>
      `;
      geotiffNotice.classList.add("hidden");
      const reader = new FileReader();
      reader.onload = (e) => {
        imagePreviewElement.src = e.target.result;
        imagePreviewElement.classList.remove("hidden");
      };
      reader.readAsDataURL(file);
    }

    filePreviewCard.classList.remove("hidden");
    metadataStatusIndicator.textContent = "File Loaded";
    metadataStatusIndicator.style.color = "var(--accent-mint)";

    metaFileName.textContent = file.name;
    metaFileName.classList.add("active");
    metaFileType.textContent = isTiff ? "GeoTIFF (image/tiff)" : (file.type || `image/${ext}`);
    metaFileType.classList.add("active");
    metaFileSize.textContent = `${formattedSize} (${file.size.toLocaleString()} bytes)`;
    metaFileSize.classList.add("active");

    if (isTiff) {
      metaImageWidth.innerHTML = `<span class="notice-text">Decoding via Rasterio...</span>`;
      metaImageHeight.innerHTML = `<span class="notice-text">Decoding via Rasterio...</span>`;
      metaNumberBands.innerHTML = `<span class="notice-text">Multi-spectral container</span>`;
      
      // Enable Action Button for GeoTIFF
      calcParamsBtn.disabled = false;
      calcParamsBtn.className = "btn-primary-active";
      calcBtnText.textContent = "Extract Features & Predict Nutrients";
      calcActionNoteText.textContent = "Click to run multi-spectral analysis & Random Forest inference.";
      
      backendStatusBadge.innerHTML = `
        <span class="status-dot"></span>
        <span class="status-text">Ready: <strong class="text-accent">${file.name}</strong></span>
      `;

      // Trigger analysis automatically for seamless UX
      triggerAnalysis(file);
    } else {
      metaImageWidth.textContent = "—";
      metaImageHeight.textContent = "—";
      metaNumberBands.innerHTML = `<span class="notice-text">3-channel RGB (Standard web raster)</span>`;
      metaSpatialRes.innerHTML = `<span class="notice-text">No geospatial coordinates</span>`;
      metaCRS.innerHTML = `<span class="notice-text">None</span>`;

      calcParamsBtn.disabled = true;
      calcParamsBtn.className = "btn-primary-disabled";
      calcBtnText.textContent = "Multi-spectral GeoTIFF Required";
      calcActionNoteText.textContent = "Analysis requires calibrated satellite bands (Red + NIR). RGB images lack NIR.";

      showError("Uploaded image is a standard 3-channel RGB image. Bio-optical nutrient analysis requires a multispectral GeoTIFF with NIR (B08).");
    }
  }

  // --------------------------------------------------------------------------
  // 5. Realistic Non-Blocking Analysis Loader (~3s minimum presentation)
  // --------------------------------------------------------------------------
  function setTimelineStep(activeKey, progressPercent) {
    const stepKeys = ["upload", "bands", "features", "npk", "groq", "complete"];
    const activeIdx = stepKeys.indexOf(activeKey);

    stepKeys.forEach((key, idx) => {
      const el = timelineSteps[key];
      if (!el) return;
      if (idx < activeIdx) {
        el.className = "timeline-step completed";
      } else if (idx === activeIdx) {
        el.className = "timeline-step active";
      } else {
        el.className = "timeline-step";
      }
    });

    if (analysisProgressBar) {
      analysisProgressBar.style.width = `${progressPercent}%`;
    }
  }

  async function triggerAnalysis(file) {
    if (!file) return;

    clearError();
    calcParamsBtn.disabled = true;
    calcBtnText.textContent = "Processing Satellite Raster...";
    backendStatusBadge.innerHTML = `
      <span class="status-dot" style="background:#f59e0b"></span>
      <span class="status-text">Analyzing: <strong style="color:#f59e0b">${file.name}</strong></span>
    `;

    // Show professional progress overlay
    if (analysisOverlay) {
      analysisOverlay.classList.remove("hidden");
    }

    const startTime = performance.now();

    // Start UI timeline progression matching analysis stages:
    // 0.0 - 0.6s: Reading satellite data...
    // 0.6 - 1.3s: Extracting spectral bands...
    // 1.3 - 2.0s: Calculating vegetation features...
    // 2.0 - 2.5s: Estimating N/P/K...
    // 2.5 - 3.0s: Generating agricultural advice...
    let currentStage = "upload";
    analysisCurrentStageText.textContent = "Reading satellite data...";
    setTimelineStep("upload", 15);

    const stageTimers = [];
    stageTimers.push(setTimeout(() => {
      currentStage = "bands";
      analysisCurrentStageText.textContent = "Extracting spectral bands...";
      setTimelineStep("bands", 35);
    }, 600));

    stageTimers.push(setTimeout(() => {
      currentStage = "features";
      analysisCurrentStageText.textContent = "Calculating vegetation features...";
      setTimelineStep("features", 60);
    }, 1300));

    stageTimers.push(setTimeout(() => {
      currentStage = "npk";
      analysisCurrentStageText.textContent = "Estimating N/P/K...";
      setTimelineStep("npk", 80);
    }, 2000));

    stageTimers.push(setTimeout(() => {
      currentStage = "groq";
      analysisCurrentStageText.textContent = "Generating agricultural advice...";
      setTimelineStep("groq", 95);
    }, 2500));

    // Prepare request payload including selected crop and growth stage
    const formData = new FormData();
    formData.append("file", file);
    if (cropSelect && cropSelect.value) {
      formData.append("crop", cropSelect.value);
    }
    if (growthStageSelect && growthStageSelect.value) {
      formData.append("growth_stage", growthStageSelect.value);
    }

    try {
      // Execute the real backend request asynchronously
      const response = await fetch("/api/analyze", {
        method: "POST",
        body: formData
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: "Unknown server error" }));
        throw new Error(errorData.detail || `Server returned HTTP ${response.status}`);
      }

      const result = await response.json();

      // Check elapsed time: ensure visible presentation targets ~3s
      // If processing finishes faster, wait only for remaining time.
      // If processing takes longer, keep loader visible until processing completes.
      const elapsed = performance.now() - startTime;
      const remainingTime = Math.max(0, MIN_DISPLAY_MS - elapsed);
      if (remainingTime > 0) {
        await new Promise((resolve) => setTimeout(resolve, remainingTime));
      }

      // Clear any pending intermediate timers
      stageTimers.forEach(clearTimeout);

      // Final completion step presentation
      analysisCurrentStageText.textContent = "Analysis complete ✓";
      setTimelineStep("complete", 100);

      // Brief transition pause so user perceives completion
      await new Promise((resolve) => setTimeout(resolve, 400));

      // Hide overlay and reveal dashboard results
      if (analysisOverlay) {
        analysisOverlay.classList.add("hidden");
      }

      applyAnalysisResults(result);

    } catch (err) {
      stageTimers.forEach(clearTimeout);
      if (analysisOverlay) {
        analysisOverlay.classList.add("hidden");
      }

      showError(`Analysis failed: ${err.message}`);
      calcParamsBtn.disabled = false;
      calcParamsBtn.className = "btn-primary-active";
      calcBtnText.textContent = "Retry Analysis";
      backendStatusBadge.innerHTML = `
        <span class="status-dot" style="background:#ef4444"></span>
        <span class="status-text">Error: <strong style="color:#ef4444">Analysis Failed</strong></span>
      `;
    }
  }

  // --------------------------------------------------------------------------
  // 6. Apply Analysis Results to UI
  // --------------------------------------------------------------------------
  function applyAnalysisResults(result) {
    const meta = result.metadata || {};
    const bands = result.bands || {};
    const features = result.features || {};
    const preds = result.predictions || {};
    const advisor = result.advisor;

    // 1. Update Retrieved Metadata Table
    if (metaImageWidth) metaImageWidth.textContent = `${meta.width} px`;
    if (metaImageHeight) metaImageHeight.textContent = `${meta.height} px`;
    if (metaNumberBands) metaNumberBands.textContent = `${meta.count} Bands`;
    if (metaSpatialRes) metaSpatialRes.textContent = meta.spatial_resolution || "10.0 m (Sentinel-2)";
    if (metaCRS) metaCRS.textContent = meta.crs || "EPSG:4326";

    // 2. Update Spectral Bands Panel
    if (bands.blue !== undefined && bandValues.blue) bandValues.blue.textContent = `${bands.blue.toFixed(4)} (Surface Reflectance)`;
    if (bands.green !== undefined && bandValues.green) bandValues.green.textContent = `${bands.green.toFixed(4)} (Surface Reflectance)`;
    if (bands.red !== undefined && bandValues.red) bandValues.red.textContent = `${bands.red.toFixed(4)} (Surface Reflectance)`;
    if (bands.nir !== undefined && bandValues.nir) bandValues.nir.textContent = `${bands.nir.toFixed(4)} (Surface Reflectance)`;
    if (bandValues.redEdge) bandValues.redEdge.textContent = "Derived / Filtered";
    if (bandValues.swir) bandValues.swir.textContent = "Optional (B11)";

    Object.values(bandValues).forEach((el) => {
      if (el) el.classList.remove("notice-text");
    });

    // 3. Update Section 1: Satellite Analysis (6 Core Features)
    if (satelliteAnalysisBadge) {
      satelliteAnalysisBadge.textContent = "Features Extracted";
      satelliteAnalysisBadge.style.color = "var(--accent-mint)";
      satelliteAnalysisBadge.style.borderColor = "rgba(74, 222, 128, 0.4)";
    }

    if (features.NDVI !== null && paramStatuses.ndvi) {
      paramStatuses.ndvi.textContent = `${features.NDVI.toFixed(4)} (Index)`;
      paramStatuses.ndvi.className = "param-status calculated";
    }

    if (features.EVI !== null && paramStatuses.evi) {
      paramStatuses.evi.textContent = `${features.EVI.toFixed(4)} (Index)`;
      paramStatuses.evi.className = "param-status calculated";
    }

    if (features.FVC !== null && paramStatuses.fvc) {
      paramStatuses.fvc.textContent = `${(features.FVC * 100).toFixed(1)}% (${features.FVC.toFixed(4)})`;
      paramStatuses.fvc.className = "param-status calculated";
    }

    if (features.LAI !== null && paramStatuses.lai) {
      paramStatuses.lai.textContent = `${features.LAI.toFixed(2)} m²/m²`;
      paramStatuses.lai.className = "param-status calculated";
    }

    if (features["NDVI Texture"] !== null && paramStatuses.texture) {
      paramStatuses.texture.textContent = `${features["NDVI Texture"].toFixed(4)} (Contrast)`;
      paramStatuses.texture.className = "param-status calculated";
    }

    if (paramStatuses.zone && features.NDVI !== null) {
      const zone = features.NDVI > 0.6 ? "High Vigor" : (features.NDVI > 0.4 ? "Moderate Vigor" : "Low / Bare Soil");
      paramStatuses.zone.textContent = zone;
      paramStatuses.zone.className = "param-status calculated";
    }

    if (features.Biomass !== null && paramStatuses.biomass) {
      paramStatuses.biomass.textContent = `${features.Biomass.toFixed(2)} (Proxy Index)`;
      paramStatuses.biomass.className = "param-status calculated";
    }

    if (paramStatuses.lst) {
      paramStatuses.lst.textContent = "Not in Sentinel-2 (Requires Landsat/S3)";
      paramStatuses.lst.className = "param-status pending";
    }

    // 4. Update Section 2: Estimated Soil Nutrients (Model Estimated N/P/K)
    if (preds && preds.N !== undefined) {
      if (predNVal) predNVal.textContent = preds.N.toFixed(2);
      if (predPVal) predPVal.textContent = preds.P.toFixed(2);
      if (predKVal) predKVal.textContent = preds.K.toFixed(2);
      if (nutrientStatusBadge) {
        nutrientStatusBadge.textContent = "Model Estimated";
        nutrientStatusBadge.style.color = "var(--accent-mint)";
        nutrientStatusBadge.style.borderColor = "rgba(74, 222, 128, 0.4)";
      }
    }

    // 5. Update Section 3: Groq AI Agricultural Advisor
    if (advisor && advisor.available && advisor.advice) {
      const adv = advisor.advice;
      if (advisorEmptyState) advisorEmptyState.classList.add("hidden");
      if (advisorFallbackBanner) advisorFallbackBanner.classList.add("hidden");
      if (advisorResults) advisorResults.classList.remove("hidden");

      // Overall Assessment
      if (advisorSummary) advisorSummary.textContent = adv.summary || "No assessment available.";

      // Nutrient Observations
      if (advisorObservationsList) {
        advisorObservationsList.innerHTML = "";
        const obsList = Array.isArray(adv.nutrient_observations) ? adv.nutrient_observations : [];
        obsList.forEach((item) => {
          const li = document.createElement("li");
          li.textContent = item;
          advisorObservationsList.appendChild(li);
        });
      }

      // Possible Concerns
      if (advisorConcernsList) {
        advisorConcernsList.innerHTML = "";
        const concerns = Array.isArray(adv.possible_concerns) ? adv.possible_concerns : [];
        concerns.forEach((item) => {
          const li = document.createElement("li");
          li.textContent = item;
          advisorConcernsList.appendChild(li);
        });
      }

      // Actions to Consider
      if (advisorActionsList) {
        advisorActionsList.innerHTML = "";
        const actions = Array.isArray(adv.actions_to_consider) ? adv.actions_to_consider : [];
        actions.forEach((item) => {
          const li = document.createElement("li");
          li.textContent = item;
          advisorActionsList.appendChild(li);
        });
      }

      // Things to Monitor
      if (advisorMonitorList) {
        advisorMonitorList.innerHTML = "";
        const monitors = Array.isArray(adv.things_to_monitor) ? adv.things_to_monitor : [];
        monitors.forEach((item) => {
          const li = document.createElement("li");
          li.textContent = item;
          advisorMonitorList.appendChild(li);
        });
      }

      // Validation Note
      if (advisorValidationNote) {
        advisorValidationNote.textContent = adv.validation_note ||
          "These nutrient values are model estimates and should be validated with appropriate soil/field observations before fertilizer application.";
      }

      if (advisorStatusBadge) {
        advisorStatusBadge.textContent = "Advisor Active (Groq AI)";
        advisorStatusBadge.style.color = "var(--accent-mint)";
        advisorStatusBadge.style.borderColor = "rgba(74, 222, 128, 0.4)";
      }
    } else {
      // Safe fallback when Groq is unavailable or backend API key is not configured
      if (advisorEmptyState) advisorEmptyState.classList.add("hidden");
      if (advisorResults) advisorResults.classList.add("hidden");
      if (advisorFallbackBanner) advisorFallbackBanner.classList.remove("hidden");

      if (advisorFallbackMsg) {
        advisorFallbackMsg.textContent = (advisor && advisor.error)
          ? advisor.error
          : "AI agricultural advice is temporarily unavailable. Nutrient estimation was completed successfully.";
      }

      if (advisorStatusBadge) {
        advisorStatusBadge.textContent = "Advisor Standby";
        advisorStatusBadge.style.color = "#f59e0b";
        advisorStatusBadge.style.borderColor = "rgba(245, 158, 11, 0.4)";
      }
    }

    // 6. Update Button & Global Status
    calcParamsBtn.disabled = false;
    calcParamsBtn.className = "btn-primary-active";
    calcBtnText.textContent = "Analysis Complete (Re-run)";
    calcActionNoteText.textContent = "Features extracted, N/P/K estimated, and agricultural advice updated.";

    backendStatusBadge.innerHTML = `
      <span class="status-dot"></span>
      <span class="status-text">Analyzed: <strong class="text-accent">${result.filename || "satellite_tile.tif"}</strong></span>
    `;
  }

  // --------------------------------------------------------------------------
  // 7. Reset Dashboard
  // --------------------------------------------------------------------------
  function resetDashboard() {
    fileInput.value = "";
    currentActiveFile = null;
    clearError();

    filePreviewCard.classList.add("hidden");
    imagePreviewElement.src = "";
    imagePreviewElement.classList.add("hidden");
    geotiffNotice.classList.add("hidden");

    if (analysisOverlay) analysisOverlay.classList.add("hidden");

    metadataStatusIndicator.textContent = "Awaiting File";
    metadataStatusIndicator.style.color = "";
    metaFileName.textContent = "—";
    metaFileName.classList.remove("active");
    metaFileType.textContent = "—";
    metaFileType.classList.remove("active");
    metaFileSize.textContent = "—";
    metaFileSize.classList.remove("active");
    metaImageWidth.textContent = "—";
    metaImageHeight.textContent = "—";
    metaNumberBands.textContent = "—";
    metaSpatialRes.textContent = "—";
    metaCRS.textContent = "—";

    Object.keys(bandValues).forEach((key) => {
      if (bandValues[key]) {
        bandValues[key].textContent = "Waiting for satellite data...";
        bandValues[key].classList.remove("notice-text");
      }
    });

    if (satelliteAnalysisBadge) {
      satelliteAnalysisBadge.textContent = "Awaiting Analysis";
      satelliteAnalysisBadge.style.color = "";
      satelliteAnalysisBadge.style.borderColor = "";
    }

    Object.keys(paramStatuses).forEach((key) => {
      if (paramStatuses[key]) {
        paramStatuses[key].textContent = "Not calculated yet";
        paramStatuses[key].className = "param-status";
      }
    });

    if (predNVal) predNVal.textContent = "—";
    if (predPVal) predPVal.textContent = "—";
    if (predKVal) predKVal.textContent = "—";
    if (nutrientStatusBadge) {
      nutrientStatusBadge.textContent = "Awaiting Analysis";
      nutrientStatusBadge.style.color = "";
      nutrientStatusBadge.style.borderColor = "";
    }

    // Reset Advisor Card
    if (advisorEmptyState) advisorEmptyState.classList.remove("hidden");
    if (advisorResults) advisorResults.classList.add("hidden");
    if (advisorFallbackBanner) advisorFallbackBanner.classList.add("hidden");
    if (advisorStatusBadge) {
      advisorStatusBadge.textContent = "Awaiting Analysis";
      advisorStatusBadge.style.color = "";
      advisorStatusBadge.style.borderColor = "";
    }

    calcParamsBtn.disabled = true;
    calcParamsBtn.className = "btn-primary-disabled";
    calcBtnText.textContent = "Extract Features & Predict Nutrients";
    calcActionNoteText.textContent = "Upload a GeoTIFF to automatically extract bands and estimate N/P/K.";

    backendStatusBadge.innerHTML = `
      <span class="status-dot"></span>
      <span class="status-text">Satellite Ingestion: <strong class="text-accent">Standby</strong></span>
    `;
  }

  // --------------------------------------------------------------------------
  // 8. Event Listeners
  // --------------------------------------------------------------------------
  calcParamsBtn.addEventListener("click", () => {
    if (currentActiveFile) {
      triggerAnalysis(currentActiveFile);
    }
  });

  // Re-run analysis if crop or growth stage changes with an active file
  if (cropSelect) {
    cropSelect.addEventListener("change", () => {
      if (currentActiveFile && !calcParamsBtn.disabled) {
        triggerAnalysis(currentActiveFile);
      }
    });
  }

  if (growthStageSelect) {
    growthStageSelect.addEventListener("change", () => {
      if (currentActiveFile && !calcParamsBtn.disabled) {
        triggerAnalysis(currentActiveFile);
      }
    });
  }

  ["dragenter", "dragover", "dragleave", "drop"].forEach((eventName) => {
    window.addEventListener(eventName, (e) => e.preventDefault(), false);
  });

  ["dragenter", "dragover"].forEach((eventName) => {
    dropzone.addEventListener(eventName, () => dropzone.classList.add("dragover"));
  });

  ["dragleave", "dragend"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (e) => {
      if (!dropzone.contains(e.relatedTarget)) dropzone.classList.remove("dragover");
    });
  });

  dropzone.addEventListener("drop", (e) => {
    dropzone.classList.remove("dragover");
    const dt = e.dataTransfer;
    if (dt && dt.files && dt.files.length > 0) {
      processFile(dt.files[0]);
    }
  });

  dropzone.addEventListener("click", (e) => {
    if (e.target !== browseBtn && e.target !== loadSampleBtn) fileInput.click();
  });

  browseBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    fileInput.click();
  });

  const loadSampleBtn = document.getElementById("loadSampleBtn");
  if (loadSampleBtn) {
    loadSampleBtn.addEventListener("click", async (e) => {
      e.stopPropagation();
      try {
        backendStatusBadge.innerHTML = `
          <span class="status-dot" style="background:#38bdf8"></span>
          <span class="status-text">Fetching: <strong style="color:#38bdf8">test_raster.tif</strong></span>
        `;
        const res = await fetch("/test_raster.tif");
        if (!res.ok) throw new Error("Could not fetch test_raster.tif from server");
        const blob = await res.blob();
        const sampleFile = new File([blob], "test_raster.tif", { type: "image/tiff" });
        processFile(sampleFile);
      } catch (err) {
        showError(`Failed to load sample: ${err.message}`);
      }
    });
  }

  dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fileInput.click();
    }
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files && fileInput.files.length > 0) {
      processFile(fileInput.files[0]);
    }
  });

  removeFileBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    resetDashboard();
  });
});
