import 'package:bloc/bloc.dart';
import 'package:security_app/features/monitoring/data/repo/monitoring_repository.dart';
import 'package:security_app/features/monitoring/logic/cubit/monitor_state.dart';

class MonitoringCubit extends Cubit<MonitoringState> {
  final MonitoringRepository monitoringRepository;

  MonitoringCubit({required this.monitoringRepository})
      : super(MonitoringInitial());

  // دالة بدء المراقبة
  Future<void> startMonitoring() async {
    try {
      emit(MonitoringLoading());  // حالة التحميل

      // تحقق من الاتصال بالإنترنت
      final isConnected = await monitoringRepository.checkInternetConnection();
      if (!isConnected) {
        emit(MonitoringError(message: 'لا يوجد اتصال بالإنترنت'));
        return;
      }

      // تحقق من الأذونات
      final hasPermissions = await monitoringRepository.checkPermissions();
      if (!hasPermissions) {
        emit(MonitoringError(message: 'الأذونات غير كافية'));
        return;
      }

      // تحقق من أن البث الفيديو متاح
      final isVideoAvailable = await monitoringRepository.isVideoFeedAvailable();
      if (!isVideoAvailable) {
        emit(MonitoringError(message: 'فشل في تحميل البث'));
        return;
      }

      // إذا تم كل شيء بنجاح
      emit(MonitoringSuccess());
    } catch (e) {
      emit(MonitoringError(message: 'حدث خطأ غير متوقع'));
    }
  }
}
