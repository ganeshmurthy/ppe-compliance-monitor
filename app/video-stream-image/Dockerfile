# Video stream publisher: downloads MP4 from MinIO and streams via RTSP to MediaMTX
FROM docker.io/minio/mc:latest AS mc

FROM alpine:3.19

RUN apk add --no-cache ffmpeg netcat-openbsd

COPY --from=mc /usr/bin/mc /usr/bin/mc

WORKDIR /stream

COPY video-stream-image/stream-from-minio.sh /stream/stream-from-minio.sh
RUN chmod +x /stream/stream-from-minio.sh

ENTRYPOINT ["/stream/stream-from-minio.sh"]
