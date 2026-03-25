{{- define "label-studio.name" -}}
{{- if .Values.global.chartName -}}
{{- .Values.global.chartName -}}
{{- else if .Values.nameOverride -}}
{{- .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- .Chart.Name -}}
{{- end -}}
{{- end -}}

{{- define "label-studio.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $release := default .Release.Name .Values.global.releaseName -}}
{{- $chartName := default "ppe-compliance-monitor" .Values.global.chartName -}}
{{- printf "%s-%s-labelstudio" $release $chartName | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}

{{- define "label-studio.labels" -}}
app.kubernetes.io/name: {{ include "label-studio.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" }}
{{- end -}}
