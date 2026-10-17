from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    mean_radius: float = Field(..., description="Mean radius", ge=0.0)
    mean_texture: float = Field(..., description="Mean texture", ge=0.0)
    mean_perimeter: float = Field(..., description="Mean perimeter", ge=0.0)
    mean_area: float = Field(..., description="Mean area", ge=0.0)
    mean_smoothness: float = Field(..., description="Mean smoothness", ge=0.0)
    mean_compactness: float = Field(..., description="Mean compactness", ge=0.0)
    mean_concavity: float = Field(..., description="Mean concavity", ge=0.0)
    mean_concave_points: float = Field(..., description="Mean concave points", ge=0.0)
    mean_symmetry: float = Field(..., description="Mean symmetry", ge=0.0)
    mean_fractal_dimension: float = Field(
        ..., description="Mean fractal dimension", ge=0.0
    )

    radius_error: float = Field(..., description="Radius standard error", ge=0.0)
    texture_error: float = Field(..., description="Texture standard error", ge=0.0)
    perimeter_error: float = Field(..., description="Perimeter standard error", ge=0.0)
    area_error: float = Field(..., description="Area standard error", ge=0.0)
    smoothness_error: float = Field(
        ..., description="Smoothness standard error", ge=0.0
    )
    compactness_error: float = Field(
        ..., description="Compactness standard error", ge=0.0
    )
    concavity_error: float = Field(..., description="Concavity standard error", ge=0.0)
    concave_points_error: float = Field(
        ..., description="Concave points standard error", ge=0.0
    )
    symmetry_error: float = Field(..., description="Symmetry standard error", ge=0.0)
    fractal_dimension_error: float = Field(
        ..., description="Fractal dimension standard error", ge=0.0
    )

    worst_radius: float = Field(..., description="Worst radius", ge=0.0)
    worst_texture: float = Field(..., description="Worst texture", ge=0.0)
    worst_perimeter: float = Field(..., description="Worst perimeter", ge=0.0)
    worst_area: float = Field(..., description="Worst area", ge=0.0)
    worst_smoothness: float = Field(..., description="Worst smoothness", ge=0.0)
    worst_compactness: float = Field(..., description="Worst compactness", ge=0.0)
    worst_concavity: float = Field(..., description="Worst concavity", ge=0.0)
    worst_concave_points: float = Field(..., description="Worst concave points", ge=0.0)
    worst_symmetry: float = Field(..., description="Worst symmetry", ge=0.0)
    worst_fractal_dimension: float = Field(
        ..., description="Worst fractal dimension", ge=0.0
    )


class PredictionResponse(BaseModel):
    predicted_class: int
    readable_label: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
