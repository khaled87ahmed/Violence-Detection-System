class ViolenceStatus {
  final bool violenceDetected;

  ViolenceStatus({required this.violenceDetected});

  factory ViolenceStatus.fromJson(Map<String, dynamic> json) {
    return ViolenceStatus(
      violenceDetected: json['violence'],
    );
  }
}
