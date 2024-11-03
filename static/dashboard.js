
function openTab(evt, tabName) {
    const tabContent = document.getElementsByClassName("tab-content");
    for (let i = 0; i < tabContent.length; i++) {
        tabContent[i].classList.remove("active");
    }

    const tabButtons = document.getElementsByClassName("tab-button");
    for (let i = 0; i < tabButtons.length; i++) {
        tabButtons[i].classList.remove("active");
    }

    document.getElementById(tabName).classList.add("active");
    evt.currentTarget.classList.add("active");
}

// Function to recommend music
async function recommendMusic() {
    const imageInput = document.getElementById("musicImage").files[0];
    const formData = new FormData();
    formData.append("musicImage", imageInput);

    const response = await fetch('/recommend-music', {
        method: 'POST',
        body: formData
    });
    const data = await response.json();
    document.getElementById("musicResult").textContent = data.song || "No recommendation found.";
}


// Function to crop image
async function cropImage() {
    const imageInput = document.getElementById("image").files[0];
    const formData = new FormData();
    formData.append("image", imageInput);

    const response = await fetch('/crop-image', {
        method: 'POST',
        body: formData
    });
    const data = await response.blob();
    const imageURL = URL.createObjectURL(data);
    const croppedImage = document.getElementById("croppedImage");
    croppedImage.src = imageURL;
    croppedImage.style.display = "block";
}

// Function to generate caption
async function generateCaption() {
    const prompt = document.getElementById("captionPrompt").value;
    const response = await fetch('/generate-caption', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt })
    });
    const data = await response.json();
    document.getElementById("captionResult").textContent = data.caption || "No caption found.";
}

// Function to show preview
function showPreview() {
    document.getElementById("previewMusic").style.display = "block";
    document.getElementById("previewImage").style.display = "block";
    document.getElementById("previewCaption").textContent = document.getElementById("captionResult").textContent;
}
