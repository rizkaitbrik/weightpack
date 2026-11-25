{{- define "weightpack.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "weightpack.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{- define "weightpack.labels" -}}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" }}
app.kubernetes.io/name: {{ include "weightpack.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/* Selector labels of one component: pass (dict "root" $ "component" "agent"). */}}
{{- define "weightpack.selectorLabels" -}}
app.kubernetes.io/name: {{ include "weightpack.name" .root }}
app.kubernetes.io/instance: {{ .root.Release.Name }}
app.kubernetes.io/component: {{ .component }}
{{- end }}

{{/* An image of the release: pass (dict "root" $ "variant" "cpu"). */}}
{{- define "weightpack.image" -}}
{{- $tag := .root.Values.image.tag | default (printf "v%s" .root.Chart.AppVersion) }}
{{- printf "%s:%s-%s" .root.Values.image.repository $tag .variant }}
{{- end }}

{{- define "weightpack.secretName" -}}
{{- .Values.auth.existingSecret | default (printf "%s-keys" (include "weightpack.fullname" .)) }}
{{- end }}

{{- define "weightpack.agentService" -}}
{{- printf "%s-agent" (include "weightpack.fullname" .) }}
{{- end }}
