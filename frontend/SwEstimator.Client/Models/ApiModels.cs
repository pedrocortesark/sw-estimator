using System.Text.Json.Serialization;

namespace SwEstimator.Client.Models;

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

[JsonConverter(typeof(JsonStringEnumConverter))]
public enum ProjectType
{
    [JsonPropertyName("mobile_app")]   MobileApp,
    [JsonPropertyName("web_saas")]     WebSaas,
    [JsonPropertyName("internal_tool")] InternalTool,
    [JsonPropertyName("data_pipeline")] DataPipeline,
}

[JsonConverter(typeof(JsonStringEnumConverter))]
public enum DetailLevel
{
    [JsonPropertyName("summary")]  Summary,
    [JsonPropertyName("medium")]   Medium,
    [JsonPropertyName("detailed")] Detailed,
}

[JsonConverter(typeof(JsonStringEnumConverter))]
public enum OutputFormat
{
    [JsonPropertyName("phases_table")] PhasesTable,
    [JsonPropertyName("line_items")]   LineItems,
    [JsonPropertyName("narrative")]    Narrative,
}

// ---------------------------------------------------------------------------
// Request
// ---------------------------------------------------------------------------

public class ReferenceProject
{
    [JsonPropertyName("name")]        public string Name        { get; set; } = "";
    [JsonPropertyName("description")] public string Description { get; set; } = "";
    [JsonPropertyName("total_hours")] public int    TotalHours  { get; set; }
    [JsonPropertyName("notes")]       public string? Notes      { get; set; }
}

public class EstimationRequest
{
    [JsonPropertyName("transcript")]          public string             Transcript         { get; set; } = "";
    [JsonPropertyName("project_type")]        public ProjectType?       ProjectType        { get; set; }
    [JsonPropertyName("detail_level")]        public DetailLevel?       DetailLevel        { get; set; }
    [JsonPropertyName("output_format")]       public OutputFormat?      OutputFormat       { get; set; }
    [JsonPropertyName("reference_projects")]  public List<ReferenceProject>? ReferenceProjects { get; set; }
}

// ---------------------------------------------------------------------------
// Response
// ---------------------------------------------------------------------------

public class UsageCost
{
    [JsonPropertyName("input_tokens")]  public int   InputTokens  { get; set; }
    [JsonPropertyName("output_tokens")] public int   OutputTokens { get; set; }
    [JsonPropertyName("total_tokens")]  public int   TotalTokens  { get; set; }
    [JsonPropertyName("cost_usd")]      public float CostUsd      { get; set; }
}

public class TaskItem
{
    [JsonPropertyName("name")]     public string Name    { get; set; } = "";
    [JsonPropertyName("hours")]    public float  Hours   { get; set; }
    [JsonPropertyName("cost_usd")] public float  CostUsd { get; set; }
}

public class Phase
{
    [JsonPropertyName("name")]           public string         Name         { get; set; } = "";
    [JsonPropertyName("tasks")]          public List<TaskItem> Tasks        { get; set; } = [];
    [JsonPropertyName("total_hours")]    public float          TotalHours   { get; set; }
    [JsonPropertyName("total_cost_usd")] public float          TotalCostUsd { get; set; }
}

public class TeamMember
{
    [JsonPropertyName("role")]       public string Role       { get; set; } = "";
    [JsonPropertyName("count")]      public int    Count      { get; set; }
    [JsonPropertyName("dedication")] public string Dedication { get; set; } = "";
}

public class EstimationResult
{
    [JsonPropertyName("executive_summary")]  public string           ExecutiveSummary { get; set; } = "";
    [JsonPropertyName("phases")]             public List<Phase>      Phases           { get; set; } = [];
    [JsonPropertyName("total_hours")]        public float            TotalHours       { get; set; }
    [JsonPropertyName("total_cost_usd")]     public float            TotalCostUsd     { get; set; }
    [JsonPropertyName("team_composition")]   public List<TeamMember> TeamComposition  { get; set; } = [];
    [JsonPropertyName("duration_weeks")]     public float            DurationWeeks    { get; set; }
    [JsonPropertyName("confidence_pct")]     public float            ConfidencePct    { get; set; } = 100f;

    public bool IsOutOfScope =>
        ExecutiveSummary.StartsWith("Out of scope:", StringComparison.OrdinalIgnoreCase);
}

public class EstimationResponse
{
    [JsonPropertyName("estimation")]     public EstimationResult Estimation    { get; set; } = new();
    [JsonPropertyName("provider_used")]  public string           ProviderUsed  { get; set; } = "";
    [JsonPropertyName("model_used")]     public string           ModelUsed     { get; set; } = "";
    [JsonPropertyName("usage")]          public UsageCost        Usage         { get; set; } = new();
    [JsonPropertyName("cached")]         public bool             Cached        { get; set; }
    [JsonPropertyName("prompt_version")] public string           PromptVersion { get; set; } = "v1";
}

// ---------------------------------------------------------------------------
// Session
// ---------------------------------------------------------------------------

public class SessionCreateResponse
{
    [JsonPropertyName("session_id")] public string SessionId { get; set; } = "";
}

public class SessionInfoResponse
{
    [JsonPropertyName("session_id")]        public string              SessionId       { get; set; } = "";
    [JsonPropertyName("turn_count")]        public int                 TurnCount       { get; set; }
    [JsonPropertyName("project_metadata")]  public Dictionary<string, object?> ProjectMetadata { get; set; } = new();
}
