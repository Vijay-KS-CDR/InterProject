/**
 * ==============================================================================
 * Agricultural AI - Satellite Image Upload & Inspection Frontend
 * ==============================================================================
 * This script handles:
 *  1. Drag-and-drop & file selection events
 *  2. File extension validation (.tif, .tiff, .png, .jpg, .jpeg)
 *  3. File metadata extraction (name, formatted size, type, image dimensions)
 *  4. Image preview generation for web-compatible formats (PNG/JPG)
 *  5. Informative placeholder updates for GeoTIFF and satellite metadata
 *  6. Resetting / removing files
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

  // Status Badge in header
  const backendStatusBadge = document.getElementById("backendStatusBadge");

  // --------------------------------------------------------------------------
  // 2. Constants & Allowed Extensions
  // --------------------------------------------------------------------------
  // Strict list of accepted file extensions per requirements
  const ALLOWED_EXTENSIONS = ["tif", "tiff", "png", "jpg", "jpeg"];
  
  // Standard text for fields that need server-side GDAL / Rasterio processing
  const BACKEND_REQUIRED_TEXT = "Not available — backend processing required";

  // --------------------------------------------------------------------------
  // 3. Helper Functions
  // --------------------------------------------------------------------------

  /**
   * Format raw bytes into human-readable units (Bytes, KB, MB)
   * Example: 1048576 -> "1.00 MB"
   */
  function formatBytes(bytes, decimals = 2) {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + " " + sizes[i];
  }

  /**
   * Extract lowercase file extension from filename
   */
  function getFileExtension(filename) {
    return filename.slice(((filename.lastIndexOf(".") - 1) >>> 0) + 2).toLowerCase();
  }

  /**
   * Validate whether the file has one of the allowed extensions
   */
  function isFileValid(file) {
    const ext = getFileExtension(file.name);
    return ALLOWED_EXTENSIONS.includes(ext);
  }

  /**
   * Show error alert message inside the dropzone
   */
  function showError(msg) {
    errorMessage.textContent = msg;
    uploadError.classList.remove("hidden");
  }

  /**
   * Hide the error message
   */
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

    // Validate file extension
    if (!isFileValid(file)) {
      const ext = getFileExtension(file.name);
      showError(`Unsupported file format (.${ext || "unknown"}). Allowed formats: GeoTIFF (.tif, .tiff), PNG, JPG/JPEG.`);
      resetDashboard();
      return;
    }

    const ext = getFileExtension(file.name);
    const isTiff = ext === "tif" || ext === "tiff";
    const formattedSize = formatBytes(file.size);

    // 1. Populate Selected File Preview Card
    previewFileName.textContent = file.name;
    previewFileSize.textContent = formattedSize;
    previewFileType.textContent = isTiff ? "GEOTIFF" : ext.toUpperCase();

    // Render appropriate SVG icon for the file type badge
    if (isTiff) {
      fileBadgeIcon.innerHTML = `
        <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="2">
          <polygon points="12 2 2 7 12 12 22 7 12 2"></polygon>
          <polyline points="2 17 12 22 22 17"></polyline>
          <polyline points="2 12 12 17 22 12"></polyline>
        </svg>
      `;
    } else {
      fileBadgeIcon.innerHTML = `
        <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="2">
          <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
          <circle cx="8.5" cy="8.5" r="1.5"></circle>
          <polyline points="21 15 16 10 5 21"></polyline>
        </svg>
      `;
    }

    // 2. Handle Visual Preview (Browser-compatible vs GeoTIFF)
    if (isTiff) {
      // Browsers cannot natively display multi-spectral 16-bit GeoTIFFs without server decoding
      imagePreviewElement.classList.add("hidden");
      imagePreviewElement.src = "";
      geotiffNotice.classList.remove("hidden");
    } else {
      // Standard PNG / JPG image preview using FileReader
      geotiffNotice.classList.add("hidden");
      const reader = new FileReader();
      reader.onload = (e) => {
        imagePreviewElement.src = e.target.result;
        imagePreviewElement.classList.remove("hidden");

        // Measure natural dimensions of the image
        const img = new Image();
        img.onload = () => {
          metaImageWidth.textContent = `${img.naturalWidth} px`;
          metaImageHeight.textContent = `${img.naturalHeight} px`;
        };
        img.src = e.target.result;
      };
      reader.readAsDataURL(file);
    }

    // Show the file preview card and update dropzone display
    filePreviewCard.classList.remove("hidden");
    metadataStatusIndicator.textContent = "File Loaded";
    metadataStatusIndicator.style.color = "var(--accent-mint)";

    // 3. Update "Retrieved Satellite Data" Panel
    metaFileName.textContent = file.name;
    metaFileName.classList.add("active");

    metaFileType.textContent = isTiff ? "GeoTIFF (image/tiff)" : (file.type || `image/${ext}`);
    metaFileType.classList.add("active");

    metaFileSize.textContent = `${formattedSize} (${file.size.toLocaleString()} bytes)`;
    metaFileSize.classList.add("active");

    if (isTiff) {
      metaImageWidth.innerHTML = `<span class="notice-text">Extracted by backend</span>`;
      metaImageHeight.innerHTML = `<span class="notice-text">Extracted by backend</span>`;
      metaNumberBands.innerHTML = `<span class="notice-text">Multi-spectral (Pending backend)</span>`;
    } else {
      // For PNG/JPG, width & height are filled asynchronously above by img.onload
      metaNumberBands.innerHTML = `<span class="notice-text">3-channel RGB (Standard web raster)</span>`;
    }

    // Explicitly enforce Requirement 12:
    // Do NOT pretend that RGB JPG/PNG contains NIR, SWIR, Red Edge, CRS, or satellite metadata.
    metaSpatialRes.innerHTML = `<span class="notice-text">${BACKEND_REQUIRED_TEXT}</span>`;
    metaCRS.innerHTML = `<span class="notice-text">${BACKEND_REQUIRED_TEXT}</span>`;

    // 4. Update "Spectral Band Values" Panel
    // Show required backend notice for all spectral bands
    Object.keys(bandValues).forEach((key) => {
      bandValues[key].textContent = BACKEND_REQUIRED_TEXT;
      bandValues[key].classList.add("notice-text");
    });

    // Update Header Status Badge
    backendStatusBadge.innerHTML = `
      <span class="status-dot"></span>
      <span class="status-text">Image Ready: <strong class="text-accent">${file.name}</strong></span>
    `;
  }

  /**
   * Reset all dashboard inputs, previews, metadata table, and band values back to initial placeholders
   */
  function resetDashboard() {
    fileInput.value = "";
    clearError();

    // Hide file preview card
    filePreviewCard.classList.add("hidden");
    imagePreviewElement.src = "";
    imagePreviewElement.classList.add("hidden");
    geotiffNotice.classList.add("hidden");

    // Reset Metadata panel to default dashes
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

    // Reset Spectral Band Values back to initial state
    Object.keys(bandValues).forEach((key) => {
      bandValues[key].textContent = "Waiting for satellite data...";
      bandValues[key].classList.remove("notice-text");
    });

    // Reset Header Badge
    backendStatusBadge.innerHTML = `
      <span class="status-dot"></span>
      <span class="status-text">Satellite Ingestion: <strong class="text-accent">Standby</strong></span>
    `;
  }

  // --------------------------------------------------------------------------
  // 5. Event Listeners: Drag & Drop
  // --------------------------------------------------------------------------

  // Prevent default drag behaviors across the window
  ["dragenter", "dragover", "dragleave", "drop"].forEach((eventName) => {
    window.addEventListener(eventName, (e) => e.preventDefault(), false);
  });

  // Highlight dropzone on dragover / dragenter
  ["dragenter", "dragover"].forEach((eventName) => {
    dropzone.addEventListener(eventName, () => {
      dropzone.classList.add("dragover");
    });
  });

  // Remove highlight on dragleave / drop
  ["dragleave", "dragend"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (e) => {
      // Only remove if leaving the dropzone boundary itself
      if (!dropzone.contains(e.relatedTarget)) {
        dropzone.classList.remove("dragover");
      }
    });
  });

  // Handle dropped file
  dropzone.addEventListener("drop", (e) => {
    dropzone.classList.remove("dragover");
    const dt = e.dataTransfer;
    if (dt && dt.files && dt.files.length > 0) {
      processFile(dt.files[0]);
    }
  });

  // --------------------------------------------------------------------------
  // 6. Event Listeners: Click to Browse
  // --------------------------------------------------------------------------

  // Clicking anywhere inside dropzone triggers native file dialog
  dropzone.addEventListener("click", (e) => {
    // If the click happened on the browse button, let browseBtn handle it
    if (e.target !== browseBtn) {
      fileInput.click();
    }
  });

  browseBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    fileInput.click();
  });

  // Keyboard accessibility: press Enter or Space to open file picker
  dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fileInput.click();
    }
  });

  // Handle file chosen via dialog
  fileInput.addEventListener("change", () => {
    if (fileInput.files && fileInput.files.length > 0) {
      processFile(fileInput.files[0]);
    }
  });

  // Remove / change file button
  removeFileBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    resetDashboard();
  });
});
