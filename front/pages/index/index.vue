<template>
	<view class="page-container">
		<view class="main-content">

			<view class="header-section">
				
				<view class="header-flex">
					<view class="logo-mark">
						<image
							class="logo-img"
							src="/static/icons/192x192.png"
							alt="301 Logo"
						/>
					</view>
					<view class="title-container">
						<view class="main-title title-gradient">
							301® Bacteria Kit
						</view>
						<view class="subtitle">
							Rapid AI Detection of  <text class="font-medium">Pathogenic Bacteria</text> 
							
							
						</view>
					</view>
				</view>
			</view>
			

			
			<view class="glass-card card-padding">
				<view class="upload-area" @tap="handleUpload">
					<image v-if="preview" :src="preview" class="big-preview" mode="aspectFill" />
					<view v-else class="upload-empty">
						<view class="upload-content">
							<p class="upload-text">Tap to capture or upload a test image</p>
						</view>
					</view>
					<view v-if="analyzing" class="analyzing-overlay">{{ statusText }}</view>
				</view>

				<view class="capture-controls">
					<h2 class="control-title">Image Capture</h2>
					<view class="button-group">
						<view class="btn-primary capture-btn" @tap="handleCapture">
							Capture
						</view>
						<view class="upload-btn" @tap="handleUpload">
							Upload
						</view>
					</view>
				</view>

				<view v-if="classifyResult" class="selection-container">
					<view class="selection-label">Detected Bacteria</view>
					<view class="protein-select">
						{{ classifyResult.bacteriaName }}
						<text v-if="classifyResult.confidence != null" class="confidence-text">
							{{ formatConfidence(classifyResult.confidence) }}
						</text>
					</view>
				</view>
			</view>

			<view class="btn-primary submit-btn" :class="{ disabled: !canSubmit }" @tap="handleSubmit">{{ analyzing ? statusText : 'Submit for Analysis' }}</view>

			<view class="modal-overlay" v-if="showModal">
				<view class="modal-content">
					<view class="modal-padding">
						<view class="modal-header">
							<h3 class="modal-title">Analysis Results</h3>
							<view id="closeModal" class="modal-close" @tap="handleCloseModal">
								<text aria-label="Close">×</text>
							</view>
						</view>

						<view class="results-container">
							<view class="result-item" v-for="r in resultList" :key="r.traceId">
								<view class="result-row">
									<text class="result-name">{{ r.biomarker }}</text>
									<text :class="r.level === 'HIGH' ? 'result-abnormal' : (r.level === 'LOW' ? 'result-abnormal' : (r.level === 'NORMAL' ? 'result-normal' : ''))">
										{{ r.concentration }} {{ r.unit }}
									</text>
								</view>
								<view v-if="r.referenceMin != null && r.referenceMax != null" class="result-reference">Reference Range: {{ r.referenceMin }} - {{ r.referenceMax }} {{ r.unit }}</view>
								<view v-if="r.warnings.length" class="error-text">{{ r.warnings.join(',') }}</view>
							</view>
						</view>

						<view class="modal-actions">
							<view id="retestBtn" class="retest-btn" @tap="handleRetest">
								Retest
							</view>
							<view id="saveBtn" class="btn-primary save-btn" @tap="handleSave">
								Save
							</view>
						</view>
					</view>
				</view>
			</view>
		</view>
	</view>
</template>

<script setup>
import { ref, computed } from 'vue'
import { classifyBacteria, calcConcentration, bacteriaNameOf } from '@/api/bacteria.js'

const filePath = ref('')
const preview = ref('')
const analyzing = ref(false)
const statusText = ref('Analyzing...')
const showModal = ref(false)
const classifyResult = ref(null)
const concentrationResult = ref(null)

const canSubmit = computed(() => !!filePath.value && !analyzing.value)

const resultList = computed(() => concentrationResult.value ? [concentrationResult.value] : [])

function formatConfidence(value) {
	const n = Number(value)
	if (Number.isNaN(n)) return ''
	return (n <= 1 ? n * 100 : n).toFixed(1) + '%'
}

function pickImage(sourceType) {
	if (analyzing.value) return
	uni.chooseImage({
		count: 1,
		sourceType,
		success(res) {
			const path = res.tempFilePaths?.[0] || res.tempFiles?.[0]?.path
			if (!path) return
			filePath.value = path
			preview.value = path
			classifyResult.value = null
			concentrationResult.value = null
			runPipeline()
		},
		fail(err) {
			console.warn('chooseImage fail', err)
		}
	})
}

const handleCapture = () => pickImage(['camera'])
const handleUpload = () => pickImage(['album'])

function toResult(classify, concentration) {
	const type = concentration.bacteriaType || classify.bacteriaType
	const name = concentration.bacteriaName || classify.bacteriaName || bacteriaNameOf(type)
	const warnings = Array.isArray(concentration.warnings) ? concentration.warnings : []
	return {
		traceId: concentration.traceId || classify.imageId || String(Date.now()),
		biomarker: name,
		concentration: concentration.concentration,
		unit: concentration.unit || 'CFU/mL',
		level: concentration.level,
		referenceMin: concentration.referenceMin,
		referenceMax: concentration.referenceMax,
		warnings
	}
}

async function runPipeline() {
	if (!filePath.value || analyzing.value) return
	analyzing.value = true
	classifyResult.value = null
	concentrationResult.value = null

	try {
		statusText.value = 'Classifying...'
		const classify = await classifyBacteria(filePath.value)
		if (!classify || !classify.bacteriaType) {
			throw new Error('Classification result is missing bacteriaType')
		}
		classifyResult.value = {
			imageId: classify.imageId,
			bacteriaType: classify.bacteriaType,
			bacteriaName: classify.bacteriaName || bacteriaNameOf(classify.bacteriaType),
			confidence: classify.confidence
		}

		statusText.value = 'Calculating...'
		const concentration = await calcConcentration({
			imageId: classify.imageId,
			bacteriaType: classify.bacteriaType
		})
		concentrationResult.value = toResult(classifyResult.value, concentration || {})
		showModal.value = true
	} catch (err) {
		uni.showToast({ title: err?.message || 'Analysis failed', icon: 'none' })
	} finally {
		analyzing.value = false
		statusText.value = 'Analyzing...'
	}
}

const handleSubmit = () => {
	if (!filePath.value) {
		uni.showToast({ title: 'Upload an image first', icon: 'none' })
		return
	}
	runPipeline()
}

const handleCloseModal = () => { showModal.value = false }
const handleRetest = () => {
	showModal.value = false
	runPipeline()
}
const handleSave = () => {
    if (!concentrationResult.value) return
    try {
        uni.setStorageSync('bacteria-analysis-result', {
            savedAt: new Date().toISOString(),
            classification: classifyResult.value,
            result: concentrationResult.value
        })
        uni.showToast({ title: 'Saved on this device', icon: 'none' })
    } catch (err) {
        uni.showToast({ title: 'Could not save result', icon: 'none' })
    }
}
</script>

<style lang="scss" scoped>

.page-container {
	min-height: 100vh;
	display: flex;
	flex-direction: column;
	font-family: "Inter", sans-serif;
	background-color: #f8fafc;
	color: #1e293b;
}

.main-content {
	padding: 16px;
	flex: 1;
	display: flex;
	flex-direction: column;
}


.header-section {
	margin-bottom: 24px;
}

.header-flex {
	display: flex;
	align-items: center;
	gap: 16px;
	flex-direction: row;
}

.logo-mark {
	width: 34px;
	height: 34px;
	border-radius: 14px;
	display: flex;
	align-items: center;
	justify-content: center;
}

.logo-img {
	width: 100%;
	height: 100%;
	object-fit: cover;
}

.title-container {
	min-width: 0;
}

.main-title {
	font-size: 26px;
	line-height: 1.25;
	font-weight: 800;
}

.title-gradient {
	background: linear-gradient(90deg, #1e293b, #334155, #4f46e5);
	-webkit-background-clip: text;
	background-clip: text;
	-webkit-text-fill-color: transparent;
	color: transparent;
}

.subtitle {
	margin-top: 4px;
	font-size: 12px;
	color: #64748b;
	white-space: normal;
}

.font-medium {
	font-weight: 500;
}


.glass-card {
	background: rgba(255, 255, 255, 0.8);
	backdrop-filter: blur(10px);
	border-radius: 12px;
	box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
	border: 1px solid rgba(255, 255, 255, 0.2);
}

.card-padding {
	padding: 16px;
	margin-bottom: 16px;
}


.protein-grid {
	display: grid;
	grid-template-columns: repeat(3, 1fr);
	gap: 8px;
	margin-bottom: 16px;
	position: relative;
}

.protein-card {
	position: relative;
	border-radius: 8px;
	margin-bottom: 16px;
	overflow: hidden;
	border: 1px solid #e5e7eb;
	box-sizing: border-box;
}

.aspect-square {
	aspect-ratio: 1/1;
}


.upload-hint {
	width: 100%;
	height: 100%;
	display: flex;
	flex-direction: column;
	align-items: center;
	justify-content: center;
	background-color: #f8fafc;
	gap: 8px;
}

.hint-text {
	font-size: 14px;
	color: #64748b;
	text-align: center;
	font-weight: 600;
}

.protein-label {
	position: absolute;
	top: 50%;
	left: 50%;
	transform: translate(-50%, -50%);
	border-radius: 6px;
	font-size: 14px;
	font-weight: 700;
	color: #4f46e5;
	text-shadow: 0 1px 3px rgba(255, 255, 255, 0.8);
	letter-spacing: 0.3px;

	white-space: normal;
	text-align: center;
	max-width: 90%;
	word-wrap: break-word;
}


.upload-area {
	position: relative;
	background-color: #f3f4f6;
	border-radius: 8px;
	margin-bottom: 16px;
	overflow: hidden;
	aspect-ratio: 1/1;
}

.upload-empty {
	position: absolute;
	top: 0;
	left: 0;
	right: 0;
	bottom: 0;
	display: flex;
	align-items: center;
	justify-content: center;
	color: #9ca3af;
}

.upload-content {
	text-align: center;
}

.upload-text {
	font-size: 14px;
}


.capture-controls {
	display: flex;
	justify-content: space-between;
	align-items: center;
	margin-bottom: 12px;
	flex-direction: row;
}

.control-title {
	font-size: 14px;
	font-weight: 600;
	color: #374151;
}

.button-group {
	display: flex;
	gap: 8px;
	flex-direction: row;
}

.capture-btn {
	padding: 6px 12px;
	border-radius: 8px;
	font-size: 14px;
	cursor: pointer;
	user-select: none;
	-webkit-user-select: none;
	-webkit-tap-highlight-color: transparent;
}

.upload-btn {
	background-color: #f3f4f6;
	padding: 6px 12px;
	border-radius: 8px;
	font-size: 14px;
	color: #374151;
	transition: background-color 0.2s;
	cursor: pointer;
	user-select: none;
	-webkit-user-select: none;
	-webkit-tap-highlight-color: transparent;
}

.upload-btn:hover {
	background-color: #e5e7eb;
}

.icon-spacing {
	margin-right: 4px;
}


.selection-container {
	margin-bottom: 12px;
}

.selection-label {
	display: block;
	font-size: 14px;
	font-weight: 500;
	color: #374151;
	margin-bottom: 4px;
}

.protein-select {
	width: 100%;
	padding: 8px;
	border: 1px solid #d1d5db;
	border-radius: 8px;
	-webkit-appearance: none;
	-moz-appearance: none;
	appearance: none;
	background-color: #ffffff;
	background-image: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 20 20'%3e%3cpath stroke='%236b7280' stroke-linecap='round' stroke-linejoin='round' stroke-width='1.5' d='M6 8l4 4 4-4'/%3e%3c/svg%3e");
	background-position: right 8px center;
	background-repeat: no-repeat;
	background-size: 24px 24px;
	font-size: 14px;
	color: #374151;
}

.protein-select:focus {
	outline: none;
	border-color: #4f46e5;
	box-shadow: 0 0 0 2px rgba(79, 70, 229, 0.2);
}

.confidence-text {
	margin-left: 8px;
	color: #64748b;
	font-weight: 500;
}


.btn-primary {
	background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
	color: white;
	transition: all 0.3s ease;
}

.btn-primary:hover {
	transform: translateY(-2px);
	box-shadow: 0 10px 15px -3px rgba(79, 70, 229, 0.3);
}

.submit-btn {
	padding: 10px 0;
	border-radius: 8px;
	font-size: 14px;
	margin-bottom: 16px;
	text-align: center;
	cursor: pointer;
	user-select: none;
	-webkit-user-select: none;
	-webkit-tap-highlight-color: transparent;
}


.modal-overlay {
	position: fixed;
	top: 0;
	left: 0;
	right: 0;
	bottom: 0;
	background-color: rgba(0, 0, 0, 0.5);
	display: flex;
	align-items: center;
	justify-content: center;
	z-index: 1000;
	transition: opacity 0.3s ease;
}

.modal-content {
	background: white;
	border-radius: 12px;
	width: 90%;
	max-width: 400px;
	transform: translateY(20px);
	transition: transform 0.3s ease;
	box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
}

.modal-padding {
	padding: 16px;
}

.modal-header {
	display: flex;
	justify-content: space-between;
	align-items: center;
	margin-bottom: 16px;
	flex-direction: row;
}

.modal-title {
	font-size: 20px;
	font-weight: 600;
	color: #1f2937;
}

.modal-close {
	color: #9ca3af;
	cursor: pointer;
	user-select: none;
	-webkit-user-select: none;
	-webkit-tap-highlight-color: transparent;
}

.modal-close:hover {
	color: #4b5563;
}


.results-container {
	display: flex;
	flex-direction: column;
	gap: 12px;
}

.result-item {
	background-color: #f9fafb;
	padding: 12px;
	border-radius: 8px;
}

.result-row {
	display: flex;
	justify-content: space-between;
	align-items: center;
	flex-direction: row;
}

.result-name {
	font-weight: 500;
}

.result-reference {
	font-size: 12px;
	color: #6b7280;
	margin-top: 4px;
}

.result-normal {
	color: #10b981;
	font-weight: 600;
}

.result-abnormal {
	color: #ef4444;
	font-weight: 600;
}


.modal-actions {
	margin-top: 16px;
	display: flex;
	justify-content: flex-end;
	gap: 8px;
	flex-direction: row;
}

.retest-btn {
	background-color: #f3f4f6;
	padding: 8px 16px;
	border-radius: 8px;
	font-size: 14px;
	color: #374151;
	transition: background-color 0.2s;
	cursor: pointer;
	user-select: none;
	-webkit-user-select: none;
	-webkit-tap-highlight-color: transparent;
}

.retest-btn:hover {
	background-color: #e5e7eb;
}

.save-btn {
	padding: 8px 16px;
	border-radius: 8px;
	font-size: 14px;
	cursor: pointer;
	user-select: none;
	-webkit-user-select: none;
	-webkit-tap-highlight-color: transparent;
}


* {
	box-sizing: border-box;
}

view {
	display: block;
}

.btn-primary,
.capture-btn,
.upload-btn,
.submit-btn,
.retest-btn,
.save-btn,
.modal-close {
	-webkit-touch-callout: none;
	-webkit-user-select: none;
	-khtml-user-select: none;
	-moz-user-select: none;
	-ms-user-select: none;
	user-select: none;
}

.protein-label,
.control-title,
.selection-label,
.modal-title,
.result-name,
.result-reference {
	-webkit-user-select: none;
	-moz-user-select: none;
	-ms-user-select: none;
	user-select: none;
}

.page-container {
	-webkit-overflow-scrolling: touch;
}

@media screen and (max-width: 375px) {
	.main-content {
		padding: 12px;
		max-width: 100%;
	}
	
	.protein-grid {
		gap: 6px;
	}
	
	.button-group {
		gap: 6px;
	}
}

@media (prefers-color-scheme: dark) {
	.page-container {
		background-color: #0f172a;
		color: #e2e8f0;
	}
	
	.glass-card {
		background: rgba(30, 41, 59, 0.8);
		border: 1px solid rgba(51, 65, 85, 0.3);
	}
}
</style>
<style lang="scss" scoped>

.protein-card .upload-badge { 
	position: absolute; 
	top: 4px; 
	left: 4px; 
	background: rgba(79,70,229,0.9); 
	color: #fff; 
	font-size: 10px; 
	padding: 2px 4px; 
	border-radius: 4px; 
}

.protein-card .thumb-preview { 
	position: absolute; 
	top: 0; 
	left: 0; 
	width: 100%; 
	height: 100%; 
	object-fit: cover; 
	filter: brightness(.9);
	padding: 2px;
	border-radius: 6px; 
}

.analyzing-overlay { 
	position: absolute; 
	top: 0; 
	left: 0; 
	width: 100%; 
	height: 100%; 
	background: rgba(0,0,0,.45); 
	display: flex; 
	align-items: center; 
	justify-content: center; 
	color: #fff; 
	font-size: 12px; 
}

.submit-btn[disabled],
.submit-btn.disabled { 
	opacity: .5; 
	filter: grayscale(.3); 
	pointer-events: none; 
}

.error-text { 
	color: #ef4444; 
	font-size: 12px; 
	margin-top: 4px; 
}

.big-preview { 
	width: 100%; 
	height: 100%; 
	object-fit: cover; 
}

.protein-card.selected { 
	box-shadow: 0 0 0 3px #4f46e5 inset;
	transform: scale(0.98);
}
</style>
