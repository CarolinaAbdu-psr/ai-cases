"""JavaScript Client SDK for AI Gateway"""

// AI Gateway JavaScript Client
class GatewayClient {
    /**
     * Initialize the gateway client
     * @param {string} baseUrl - Base URL of the gateway API (default: http://localhost:8000)
     * @param {number} timeout - Request timeout in milliseconds (default: 120000)
     */
    constructor(baseUrl = 'http://localhost:8000', timeout = 120000) {
        this.baseUrl = baseUrl.replace(/\/$/, '');
        this.timeout = timeout;
    }

    /**
     * Make a request to the gateway API
     * @private
     */
    async _request(method, path, body = null) {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), this.timeout);

        try {
            const options = {
                method,
                headers: {
                    'Content-Type': 'application/json',
                },
                signal: controller.signal,
            };

            if (body) {
                options.body = JSON.stringify(body);
            }

            const response = await fetch(`${this.baseUrl}${path}`, options);
            clearTimeout(timeoutId);

            if (!response.ok) {
                const error = await response.json().catch(() => ({ error: 'Unknown error' }));
                throw new Error(error.detail || error.error || `HTTP ${response.status}`);
            }

            // Handle 204 No Content
            if (response.status === 204) {
                return null;
            }

            return await response.json();
        } catch (error) {
            clearTimeout(timeoutId);
            if (error.name === 'AbortError') {
                throw new Error('Request timeout');
            }
            throw error;
        }
    }

    /**
     * Check gateway health
     * @returns {Promise<Object>} Health status
     */
    async healthCheck() {
        return this._request('GET', '/');
    }

    /**
     * Create a new chat session
     * @param {string} agentType - Type of agent (factory, psrio, knowledge_hub)
     * @param {string} model - Model name to use
     * @param {string} language - Chat language (default: en)
     * @param {string} userId - Optional user identifier
     * @param {Object} metadata - Optional metadata
     * @returns {Promise<Object>} Session details
     */
    async createSession(agentType, model, language = 'en', userId = null, metadata = null) {
        const request = {
            agent_type: agentType,
            model,
            language,
            user_id: userId,
            metadata,
        };
        return this._request('POST', '/sessions', request);
    }

    /**
     * Send a message in a session
     * @param {string} sessionId - Session identifier
     * @param {string} message - User message
     * @param {boolean} stream - Enable streaming (not yet implemented)
     * @param {Array} files - Optional array of file attachments
     * @returns {Promise<Object>} AI response
     */
    async chat(sessionId, message, stream = false, files = null) {
        const request = {
            session_id: sessionId,
            message,
            stream,
        };

        if (files && files.length > 0) {
            request.files = files;
        }

        return this._request('POST', '/chat', request);
    }

    /**
     * Get session information
     * @param {string} sessionId - Session identifier
     * @returns {Promise<Object>} Session details
     */
    async getSession(sessionId) {
        return this._request('GET', `/sessions/${sessionId}`);
    }

    /**
     * List all active sessions
     * @returns {Promise<Array>} List of sessions
     */
    async listSessions() {
        return this._request('GET', '/sessions');
    }

    /**
     * Delete a session
     * @param {string} sessionId - Session identifier
     * @returns {Promise<boolean>} True if deleted successfully
     */
    async deleteSession(sessionId) {
        await this._request('DELETE', `/sessions/${sessionId}`);
        return true;
    }

    /**
     * Helper function to read a file and convert to base64
     * @param {File} file - File object from input
     * @returns {Promise<Object>} File attachment object
     */
    async readFileAsBase64(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();

            reader.onload = () => {
                const base64Content = reader.result.split(',')[1]; // Remove data URL prefix
                resolve({
                    name: file.name,
                    content: base64Content,
                    mime_type: file.type || 'application/octet-stream',
                    size: file.size
                });
            };

            reader.onerror = () => reject(reader.error);
            reader.readAsDataURL(file);
        });
    }

    /**
     * Helper function to process multiple files
     * @param {FileList|Array<File>} files - Files to process
     * @returns {Promise<Array<Object>>} Array of file attachment objects
     */
    async processFiles(files) {
        const fileArray = Array.from(files);
        const promises = fileArray.map(file => this.readFileAsBase64(file));
        return Promise.all(promises);
    }
}

// Export for Node.js and browser
if (typeof module !== 'undefined' && module.exports) {
    module.exports = GatewayClient;
}

