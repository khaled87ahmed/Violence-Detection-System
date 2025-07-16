import 'package:go_router/go_router.dart';
import 'package:security_app/features/Auth/presentation/views/loginview.dart';
import 'package:security_app/features/Auth/presentation/views/registerview.dart';
import 'package:security_app/features/analyze_video/presentation/analyze_video.dart';
import 'package:security_app/features/home/presentation/views/home_view.dart';
import 'package:security_app/features/monitoring/presentation/views/monitor_view.dart';
import 'package:security_app/features/splash/presentation/views/splash_view.dart';

abstract class AppRouter {
  static const kloginView = '/Loginview';
  static const kRegisterview = '/Registerview';
  static const kHomeview = '/Homeview';
  static const kMonitorView = '/MonitorView';
  static const kAnalyzeVideoView = '/AnalyzeVideoView';
  static final router = GoRouter(
    routes: [
      GoRoute(
        path: '/',
        builder: (context, state) => const SplashView(),
      ),
      GoRoute(
        path: kloginView,
        builder: (context, state) => Loginview(),
      ),
      GoRoute(
        path: kRegisterview,
        builder: (context, state) => Registerview(),
      ),
      GoRoute(
        path: kHomeview,
        builder: (context, state) => Homeview(),
      ),
      GoRoute(
        path: kMonitorView,
        builder: (context, state) => MonitorView(),
      ),
      GoRoute(
        path: kAnalyzeVideoView,
        builder: (context, state) => const AnalyzeVideoView(),
      ),
    ],
  );
}
