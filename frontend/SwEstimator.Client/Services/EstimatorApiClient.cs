using System.Net.Http.Json;
using System.Text.Json;
using SwEstimator.Client.Models;

namespace SwEstimator.Client.Services;

/// <summary>
/// Typed HTTP client for the SW Estimator FastAPI backend.
/// All methods map 1-to-1 to an API endpoint.
/// </summary>
public class EstimatorApiClient
{
    private readonly HttpClient _http;

    private static readonly JsonSerializerOptions _jsonOptions = new()
    {
        PropertyNameCaseInsensitive = true,
    };

    public EstimatorApiClient(HttpClient http)
    {
        _http = http;
    }

    // -------------------------------------------------------------------------
    // Sessions
    // -------------------------------------------------------------------------

    /// <summary>POST /api/v1/sessions — creates a new session and returns its id.</summary>
    public async Task<SessionCreateResponse> CreateSessionAsync()
    {
        var response = await _http.PostAsync("/api/v1/sessions", null);
        response.EnsureSuccessStatusCode();
        return await response.Content.ReadFromJsonAsync<SessionCreateResponse>(_jsonOptions)
               ?? throw new InvalidOperationException("Empty response from CreateSession.");
    }

    /// <summary>GET /api/v1/sessions/{sessionId} — returns session metadata and turn count.</summary>
    public async Task<SessionInfoResponse> GetSessionAsync(string sessionId)
    {
        var response = await _http.GetAsync($"/api/v1/sessions/{sessionId}");
        response.EnsureSuccessStatusCode();
        return await response.Content.ReadFromJsonAsync<SessionInfoResponse>(_jsonOptions)
               ?? throw new InvalidOperationException("Empty response from GetSession.");
    }

    // -------------------------------------------------------------------------
    // Estimation (session-scoped, multipart/form-data)
    // -------------------------------------------------------------------------

    /// <summary>
    /// POST /api/v1/sessions/{sessionId}/estimate — sends a transcript and optional
    /// attachments as multipart/form-data and returns a structured EstimationResponse.
    /// </summary>
    /// <param name="sessionId">Active session identifier.</param>
    /// <param name="transcript">Meeting transcript or project description.</param>
    /// <param name="attachments">Optional list of (fileName, content, mimeType) tuples.</param>
    public async Task<EstimationResponse> EstimateAsync(
        string sessionId,
        string transcript,
        IEnumerable<(string FileName, byte[] Content, string MimeType)>? attachments = null)
    {
        using var form = new MultipartFormDataContent();
        form.Add(new StringContent(transcript), "transcript");

        if (attachments is not null)
        {
            foreach (var (fileName, content, mimeType) in attachments)
            {
                var fileContent = new ByteArrayContent(content);
                fileContent.Headers.ContentType =
                    new System.Net.Http.Headers.MediaTypeHeaderValue(mimeType);
                form.Add(fileContent, "attachments", fileName);
            }
        }

        var response = await _http.PostAsync($"/api/v1/sessions/{sessionId}/estimate", form);

        if (!response.IsSuccessStatusCode)
        {
            var body = await response.Content.ReadAsStringAsync();
            var detail = TryExtractDetail(body);
            throw response.StatusCode switch
            {
                System.Net.HttpStatusCode.BadRequest =>
                    new GuardrailException(detail ?? "Input rejected by guardrail."),
                System.Net.HttpStatusCode.BadGateway =>
                    new UpstreamLlmException(detail ?? "LLM upstream error."),
                _ => new HttpRequestException($"HTTP {(int)response.StatusCode}: {detail ?? body}")
            };
        }

        return await response.Content.ReadFromJsonAsync<EstimationResponse>(_jsonOptions)
               ?? throw new InvalidOperationException("Empty response from Estimate.");
    }

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    private static string? TryExtractDetail(string json)
    {
        try
        {
            using var doc = JsonDocument.Parse(json);
            if (doc.RootElement.TryGetProperty("detail", out var detail))
                return detail.GetString();
        }
        catch { /* not JSON — return null */ }
        return null;
    }
}

// ---------------------------------------------------------------------------
// Domain exceptions (mirror Python GuardrailError / UpstreamError)
// ---------------------------------------------------------------------------

public class GuardrailException : Exception
{
    public GuardrailException(string message) : base(message) { }
}

public class UpstreamLlmException : Exception
{
    public UpstreamLlmException(string message) : base(message) { }
}
