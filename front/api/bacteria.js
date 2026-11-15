import { BASE_URL } from './config.js'

const BACTERIA_NAME = {
	ECOLI: 'Escherichia coli',
	VIBRIO_PARA: 'Vibrio parahaemolyticus',
	VIBRIO_ALG: 'Vibrio alginolyticus'
}

function parseBody(data) {
	if (data == null || data === '') throw new Error('Empty response')
	if (typeof data === 'object') return data
	try {
		return JSON.parse(data)
	} catch (e) {
		throw new Error('Response is not valid JSON')
	}
}

function unwrap(body) {
	if (body && typeof body.code !== 'undefined' && body.code !== 0 && body.code !== 200) {
		throw new Error(body.message || body.msg || 'Request failed')
	}
	return body && typeof body.data !== 'undefined' ? body.data : body
}

function request(url, options = {}) {
	return new Promise((resolve, reject) => {
		uni.request({
			url: BASE_URL + url,
			method: options.method || 'POST',
			data: options.data,
			header: {
				'Content-Type': 'application/json',
				...(options.header || {})
			},
			success(res) {
				if (res.statusCode < 200 || res.statusCode >= 300) {
					reject(new Error('HTTP ' + res.statusCode))
					return
				}
				try {
					resolve(unwrap(parseBody(res.data)))
				} catch (err) {
					reject(err)
				}
			},
			fail(err) {
				reject(new Error((err && err.errMsg) || 'Network error'))
			}
		})
	})
}

function upload(url, filePath, formData = {}) {
	return new Promise((resolve, reject) => {
		uni.uploadFile({
			url: BASE_URL + url,
			filePath,
			name: 'file',
			formData,
			success(res) {
				if (res.statusCode < 200 || res.statusCode >= 300) {
					reject(new Error('HTTP ' + res.statusCode))
					return
				}
				try {
					resolve(unwrap(parseBody(res.data)))
				} catch (err) {
					reject(err)
				}
			},
			fail(err) {
				reject(new Error((err && err.errMsg) || 'Upload failed'))
			}
		})
	})
}

export function bacteriaNameOf(type) {
	return BACTERIA_NAME[type] || type || 'Unknown'
}


export function classifyBacteria(filePath) {
	return upload('/api/bacteria/classify', filePath)
}


export function calcConcentration(payload) {
	return request('/api/bacteria/concentration', {
		method: 'POST',
		data: {
			imageId: payload.imageId,
			bacteriaType: payload.bacteriaType
		}
	})
}
